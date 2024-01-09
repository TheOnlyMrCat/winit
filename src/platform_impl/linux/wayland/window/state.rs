//! The state of the window, which is shared with the event-loop.

use std::num::NonZeroU32;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use log::{error, info, warn};

use sctk::reexports::client::protocol::wl_seat::WlSeat;
use sctk::reexports::client::protocol::wl_shm::WlShm;
use sctk::reexports::client::protocol::wl_surface::WlSurface;
use sctk::reexports::client::{Connection, Proxy, QueueHandle};
use sctk::reexports::csd_frame::{
    DecorationsFrame, FrameAction, FrameClick, ResizeEdge, WindowState as XdgWindowState,
};
use sctk::reexports::protocols::wp::fractional_scale::v1::client::wp_fractional_scale_v1::WpFractionalScaleV1;
use sctk::reexports::protocols::wp::text_input::zv3::client::zwp_text_input_v3::ZwpTextInputV3;
use sctk::reexports::protocols::wp::viewporter::client::wp_viewport::WpViewport;
use sctk::reexports::protocols::xdg::shell::client::xdg_toplevel::ResizeEdge as XdgResizeEdge;

use sctk::compositor::{CompositorState, Region, SurfaceData, SurfaceDataExt};
use sctk::seat::pointer::{PointerDataExt, ThemedPointer};
use sctk::shell::wlr_layer::{LayerSurface, LayerSurfaceConfigure};
use sctk::shell::xdg::window::{DecorationMode, Window, WindowConfigure};
use sctk::shell::xdg::XdgSurface;
use sctk::shell::WaylandSurface;
use sctk::shm::slot::SlotPool;
use sctk::shm::Shm;
use sctk::subcompositor::SubcompositorState;
use wayland_protocols_plasma::blur::client::org_kde_kwin_blur::OrgKdeKwinBlur;

use crate::cursor::CursorImage;
use crate::dpi::{LogicalPosition, LogicalSize, PhysicalSize, Size};
use crate::error::{ExternalError, NotSupportedError};
use crate::event::WindowEvent;
use crate::platform_impl::wayland::event_loop::sink::EventSink;
use crate::platform_impl::wayland::types::cursor::{CustomCursor, SelectedCursor};
use crate::platform_impl::wayland::types::kwin_blur::KWinBlurManager;
use crate::platform_impl::wayland::{logical_to_physical_rounded, make_wid};
use crate::platform_impl::WindowId;
use crate::window::{CursorGrabMode, CursorIcon, ImePurpose, ResizeDirection, Theme};

use crate::platform_impl::wayland::seat::{
    PointerConstraintsState, WinitPointerData, WinitPointerDataExt, ZwpTextInputV3Ext,
};
use crate::platform_impl::wayland::state::{WindowCompositorUpdate, WinitState};

#[cfg(feature = "sctk-adwaita")]
pub type WinitFrame = sctk_adwaita::AdwaitaFrame<WinitState>;
#[cfg(not(feature = "sctk-adwaita"))]
pub type WinitFrame = sctk::shell::xdg::fallback_frame::FallbackFrame<WinitState>;

// Minimum window inner size.
const MIN_WINDOW_SIZE: LogicalSize<u32> = LogicalSize::new(2, 1);

/// The state of the window which is being updated from the [`WinitState`].
pub struct WindowState {
    /// The connection to Wayland server.
    pub connection: Connection,

    /// The `Shm` to set cursor.
    pub shm: WlShm,

    // A shared pool where to allocate custom cursors.
    custom_cursor_pool: Arc<Mutex<SlotPool>>,

    /// The pointers observed on the window.
    pub pointers: Vec<Weak<ThemedPointer<WinitPointerData>>>,

    selected_cursor: SelectedCursor,

    /// Wether the cursor is visible.
    pub cursor_visible: bool,

    /// Pointer constraints to lock/confine pointer.
    pub pointer_constraints: Option<Arc<PointerConstraintsState>>,

    /// Queue handle.
    pub queue_handle: QueueHandle<WinitState>,

    /// State that differes based on being an XDG shell or a WLR layer shell
    shell_specific: ShellSpecificState,

    /// Theme varaint.
    theme: Option<Theme>,

    /// The current window title.
    title: String,

    /// Whether the window has focus.
    has_focus: bool,

    /// The scale factor of the window.
    scale_factor: f64,

    /// Whether the window is transparent.
    transparent: bool,

    /// The state of the compositor to create WlRegions.
    compositor: Arc<CompositorState>,

    /// The current cursor grabbing mode.
    cursor_grab_mode: GrabState,

    /// Whether the IME input is allowed for that window.
    ime_allowed: bool,

    /// The current IME purpose.
    ime_purpose: ImePurpose,

    /// The text inputs observed on the window.
    text_inputs: Vec<ZwpTextInputV3>,

    /// The inner size of the window, as in without client side decorations.
    size: LogicalSize<u32>,

    /// Initial window size provided by the user. Removed on the first
    /// configure.
    initial_size: Option<Size>,

    viewport: Option<WpViewport>,
    fractional_scale: Option<WpFractionalScaleV1>,
    blur: Option<OrgKdeKwinBlur>,
    blur_manager: Option<KWinBlurManager>,
}

enum ShellSpecificState {
    Xdg {
        /// The last received configure.
        last_configure: Option<WindowConfigure>,

        /// Whether the frame is resizable.
        resizable: bool,

        /// The window frame, which is created from the configure request.
        frame: Option<WinitFrame>,

        /// Whether the CSD fail to create, so we don't try to create them on each iteration.
        csd_fails: bool,

        /// Whether we should decorate the frame.
        decorate: bool,

        /// Min size.
        min_inner_size: LogicalSize<u32>,
        max_inner_size: Option<LogicalSize<u32>>,

        /// The size of the window when no states were applied to it. The primary use for it
        /// is to fallback to original window size, before it was maximized, if the compositor
        /// sends `None` for the new size in the configure.
        stateless_size: LogicalSize<u32>,

        /// The state of the frame callback.
        frame_callback_state: FrameCallbackState,

        /// Whether the client side decorations have pending move operations.
        ///
        /// The value is the serial of the event triggered moved.
        has_pending_move: Option<u32>,

        /// The underlying SCTK window.
        window: Window,
    },
    WlrLayer {
        surface: LayerSurface,

        last_configure: Option<LayerSurfaceConfigure>,
    },
}

impl WindowState {
    /// Apply closure on the given pointer.
    fn apply_on_poiner<F: Fn(&ThemedPointer<WinitPointerData>, &WinitPointerData)>(
        &self,
        callback: F,
    ) {
        self.pointers
            .iter()
            .filter_map(Weak::upgrade)
            .for_each(|pointer| {
                let data = pointer.pointer().winit_data();
                callback(pointer.as_ref(), data);
            })
    }

    fn wl_surface(&self) -> &WlSurface {
        match &self.shell_specific {
            ShellSpecificState::Xdg { window, .. } => window.wl_surface(),
            ShellSpecificState::WlrLayer { surface, .. } => surface.wl_surface(),
        }
    }

    /// Get the current state of the frame callback.
    pub fn frame_callback_state(&self) -> FrameCallbackState {
        match self.shell_specific {
            ShellSpecificState::Xdg { frame_callback_state, .. } => frame_callback_state,
            ShellSpecificState::WlrLayer { .. } => FrameCallbackState::None,
        }
    }

    /// The frame callback was received, but not yet sent to the user.
    pub fn frame_callback_received(&mut self) {
        match &mut self.shell_specific {
            ShellSpecificState::Xdg { frame_callback_state, .. } => {
                *frame_callback_state = FrameCallbackState::Received;
            }
            ShellSpecificState::WlrLayer { .. } => {}
        }
    }

    /// Reset the frame callbacks state.
    pub fn frame_callback_reset(&mut self) {
        match &mut self.shell_specific {
            ShellSpecificState::Xdg { frame_callback_state, .. } => {
                *frame_callback_state = FrameCallbackState::None;
            }
            ShellSpecificState::WlrLayer { .. } => {},
        }
    }

    /// Request a frame callback if we don't have one for this window in flight.
    pub fn request_frame_callback(&mut self) {

        match &mut self.shell_specific {
            ShellSpecificState::Xdg { window, frame_callback_state, .. } => {
                match frame_callback_state {
                    FrameCallbackState::None | FrameCallbackState::Received => {
                        *frame_callback_state = FrameCallbackState::Requested;
                        let surface = window.wl_surface();
                        surface.frame(&self.queue_handle, surface.clone());
                    }
                    FrameCallbackState::Requested => (),
                }
            }
            ShellSpecificState::WlrLayer { .. } => {},
        }
    }

    pub fn configure_xdg(
        &mut self,
        configure: WindowConfigure,
        shm: &Shm,
        subcompositor: &Option<Arc<SubcompositorState>>,
        event_sink: &mut EventSink,
    ) -> bool {
        let ShellSpecificState::Xdg {
            ref window,
            ref mut last_configure,
            ref mut frame,
            ref mut csd_fails,
            decorate,
            ref mut stateless_size,
            ..
        } = self.shell_specific else {
            error!("configure_xdg called in layer_shell context");
            return false
        };

        // NOTE: when using fractional scaling or wl_compositor@v6 the scaling
        // should be delivered before the first configure, thus apply it to
        // properly scale the physical sizes provided by the users.
        if let Some(initial_size) = self.initial_size.take() {
            self.size = initial_size.to_logical(self.scale_factor);
            *stateless_size = self.size;
        }

        if let Some(subcompositor) = subcompositor.as_ref().filter(|_| {
            configure.decoration_mode == DecorationMode::Client
                && frame.is_none()
                && !*csd_fails
        }) {
            match WinitFrame::new(
                window,
                shm,
                #[cfg(feature = "sctk-adwaita")]
                self.compositor.clone(),
                subcompositor.clone(),
                self.queue_handle.clone(),
                #[cfg(feature = "sctk-adwaita")]
                into_sctk_adwaita_config(self.theme),
            ) {
                Ok(mut winit_frame) => {
                    winit_frame.set_title(&self.title);
                    winit_frame.set_scaling_factor(self.scale_factor);
                    // Hide the frame if we were asked to not decorate.
                    winit_frame.set_hidden(!decorate);
                    *frame = Some(winit_frame);
                }
                Err(err) => {
                    warn!("Failed to create client side decorations frame: {err}");
                    *csd_fails = true;
                }
            }
        } else if configure.decoration_mode == DecorationMode::Server {
            // Drop the frame for server side decorations to save resources.
            *frame = None;
        }

        let stateless = Self::is_stateless(&configure);

        // Emit `Occluded` event on suspension change.
        let occluded = configure.state.contains(XdgWindowState::SUSPENDED);
        if last_configure
            .as_ref()
            .map(|c| c.state.contains(XdgWindowState::SUSPENDED))
            .unwrap_or(false)
            != occluded
        {
            let window_id = make_wid(window.wl_surface());
            event_sink.push_window_event(WindowEvent::Occluded(occluded), window_id);
        }

        let (mut new_size, constrain) = if let Some(frame) = frame.as_mut() {
            // Configure the window states.
            frame.update_state(configure.state);

            match configure.new_size {
                (Some(width), Some(height)) => {
                    let (width, height) = frame.subtract_borders(width, height);
                    let width = width.map(|w| w.get()).unwrap_or(1);
                    let height = height.map(|h| h.get()).unwrap_or(1);
                    ((width, height).into(), false)
                }
                (_, _) if stateless => (*stateless_size, true),
                _ => (self.size, true),
            }
        } else {
            match configure.new_size {
                (Some(width), Some(height)) => ((width.get(), height.get()).into(), false),
                _ if stateless => (*stateless_size, true),
                _ => (self.size, true),
            }
        };

        // Apply configure bounds only when compositor let the user decide what size to pick.
        if constrain {
            let bounds = Self::inner_size_bounds(&frame, &configure);
            new_size.width = bounds
                .0
                .map(|bound_w| new_size.width.min(bound_w.get()))
                .unwrap_or(new_size.width);
            new_size.height = bounds
                .1
                .map(|bound_h| new_size.height.min(bound_h.get()))
                .unwrap_or(new_size.height);
        }

        let new_state = configure.state;
        let old_state = last_configure
            .as_ref()
            .map(|configure| configure.state);

        let state_change_requires_resize = old_state
            .map(|old_state| {
                !old_state
                    .symmetric_difference(new_state)
                    .difference(XdgWindowState::ACTIVATED | XdgWindowState::SUSPENDED)
                    .is_empty()
            })
            // NOTE: `None` is present for the initial configure, thus we must always resize.
            .unwrap_or(true);

        // NOTE: Set the configure before doing a resize, since we query it during it.
        *last_configure = Some(configure);

        if state_change_requires_resize || new_size != self.inner_size() {
            self.resize(new_size);
            true
        } else {
            false
        }
    }

    pub fn configure_layer(&mut self, configure: LayerSurfaceConfigure) -> bool {
        let ShellSpecificState::WlrLayer {
            last_configure,
            ..
        } = &mut self.shell_specific else {
            error!("configure_layer called in xdg context");
            return true;
        };

        let new_size = match configure.new_size {
            (0, 0) => self.size,
            (0, height) => (self.size.width, height).into(),
            (width, 0) => (width, self.size.height).into(),
            (width, height) => (width, height).into(),
        };

        // NOTE: Set the configure before doing a resize, since we query it during it.
        *last_configure = Some(configure);
        self.resize(new_size);

        true
    }

    /// Compute the bounds for the inner size of the surface.
    fn inner_size_bounds(
        frame: &Option<WinitFrame>,
        configure: &WindowConfigure,
    ) -> (Option<NonZeroU32>, Option<NonZeroU32>) {
        let configure_bounds = match configure.suggested_bounds {
            Some((width, height)) => (NonZeroU32::new(width), NonZeroU32::new(height)),
            None => (None, None),
        };

        if let Some(frame) = frame.as_ref() {
            let (width, height) = frame.subtract_borders(
                configure_bounds.0.unwrap_or(NonZeroU32::new(1).unwrap()),
                configure_bounds.1.unwrap_or(NonZeroU32::new(1).unwrap()),
            );
            (
                configure_bounds.0.and(width),
                configure_bounds.1.and(height),
            )
        } else {
            configure_bounds
        }
    }

    #[inline]
    fn is_stateless(configure: &WindowConfigure) -> bool {
        !(configure.is_maximized() || configure.is_fullscreen() || configure.is_tiled())
    }

    /// Start interacting drag resize.
    pub fn drag_resize_window(&self, direction: ResizeDirection) -> Result<(), ExternalError> {
        match &self.shell_specific {
            ShellSpecificState::Xdg { window, .. } => {
                let xdg_toplevel = window.xdg_toplevel();

                // TODO(kchibisov) handle touch serials.
                self.apply_on_poiner(|_, data| {
                    let serial = data.latest_button_serial();
                    let seat = data.seat();
                    xdg_toplevel.resize(seat, serial, direction.into());
                });
            }
            ShellSpecificState::WlrLayer { .. } => {}
        }

        Ok(())
    }

    /// Start the window drag.
    pub fn drag_window(&self) -> Result<(), ExternalError> {
        match &self.shell_specific {
            ShellSpecificState::Xdg { window, .. } => {
                let xdg_toplevel = window.xdg_toplevel();
                // TODO(kchibisov) handle touch serials.
                self.apply_on_poiner(|_, data| {
                    let serial = data.latest_button_serial();
                    let seat = data.seat();
                    xdg_toplevel._move(seat, serial);
                });
            }
            ShellSpecificState::WlrLayer { .. } => {} // TODO(theonlymrcat): This match should be replaced with let...else
        }

        Ok(())
    }

    /// Tells whether the window should be closed.
    #[allow(clippy::too_many_arguments)]
    pub fn frame_click(
        &mut self,
        click: FrameClick,
        pressed: bool,
        seat: &WlSeat,
        serial: u32,
        timestamp: Duration,
        window_id: WindowId,
        updates: &mut Vec<WindowCompositorUpdate>,
    ) -> Option<bool> {
        match &mut self.shell_specific {
            ShellSpecificState::Xdg { window, frame, has_pending_move, .. } => {
                match frame.as_mut()?.on_click(timestamp, click, pressed)? {
                    FrameAction::Minimize => window.set_minimized(),
                    FrameAction::Maximize => window.set_maximized(),
                    FrameAction::UnMaximize => window.unset_maximized(),
                    FrameAction::Close => WinitState::queue_close(updates, window_id),
                    FrameAction::Move => *has_pending_move = Some(serial),
                    FrameAction::Resize(edge) => {
                        let edge = match edge {
                            ResizeEdge::None => XdgResizeEdge::None,
                            ResizeEdge::Top => XdgResizeEdge::Top,
                            ResizeEdge::Bottom => XdgResizeEdge::Bottom,
                            ResizeEdge::Left => XdgResizeEdge::Left,
                            ResizeEdge::TopLeft => XdgResizeEdge::TopLeft,
                            ResizeEdge::BottomLeft => XdgResizeEdge::BottomLeft,
                            ResizeEdge::Right => XdgResizeEdge::Right,
                            ResizeEdge::TopRight => XdgResizeEdge::TopRight,
                            ResizeEdge::BottomRight => XdgResizeEdge::BottomRight,
                            _ => return None,
                        };
                        window.resize(seat, serial, edge);
                    }
                    FrameAction::ShowMenu(x, y) => window.show_window_menu(seat, serial, (x, y)),
                    _ => (),
                };
            }
            ShellSpecificState::WlrLayer { .. } => {} // TODO(theonlymrcat): This match should be replaced with let...else
        }

        Some(false)
    }

    pub fn frame_point_left(&mut self) {
        match &mut self.shell_specific {
            ShellSpecificState::Xdg { ref mut frame, .. } => {
                if let Some(frame) = frame.as_mut() {
                    frame.click_point_left();
                }
            }
            ShellSpecificState::WlrLayer { .. } => {} // TODO(theonlymrcat): This match should be replaced with let...else
        }
    }

    // Move the point over decorations.
    pub fn frame_point_moved(
        &mut self,
        seat: &WlSeat,
        surface: &WlSurface,
        timestamp: Duration,
        x: f64,
        y: f64,
    ) -> Option<CursorIcon> {
        match &mut self.shell_specific {
            ShellSpecificState::Xdg { window, frame, has_pending_move, .. } => {
                // Take the serial if we had any, so it doesn't stick around.
                let serial = has_pending_move.take();

                if let Some(frame) = frame.as_mut() {
                    let cursor = frame.click_point_moved(timestamp, &surface.id(), x, y);
                    // If we have a cursor change, that means that cursor is over the decorations,
                    // so try to apply move.
                    if let Some(serial) = cursor.is_some().then_some(serial).flatten() {
                        window.move_(seat, serial);
                        None
                    } else {
                        cursor
                    }
                } else {
                    None
                }
            }
            ShellSpecificState::WlrLayer { .. } => None
        }
    }

    /// Get the stored resizable state.
    #[inline]
    pub fn resizable(&self) -> bool {
        match self.shell_specific {
            ShellSpecificState::Xdg { resizable, .. } => resizable,
            ShellSpecificState::WlrLayer { .. } => false,
        }
    }

    /// Set the resizable state on the window.
    #[inline]
    pub fn set_resizable(&mut self, resizable: bool) {
        match &mut self.shell_specific {
            ShellSpecificState::Xdg {
                resizable: state_resizable,
                ..
            } => {
                if *state_resizable == resizable {
                    return;
                }

                *state_resizable = resizable;
            }
            ShellSpecificState::WlrLayer { .. } => {
                if resizable {
                    warn!("Resizable is ignored for layer_shell windows");
                }
                return;
            }
        }

        if resizable {
            // Restore min/max sizes of the window.
            self.reload_min_max_hints();
        } else {
            self.set_min_inner_size(Some(self.size));
            self.set_max_inner_size(Some(self.size));
        }

        // Reload the state on the frame as well.
        match &mut self.shell_specific {
            ShellSpecificState::Xdg {
                frame: Some(frame), ..
            } => {
                frame.set_resizable(resizable);
            }
            ShellSpecificState::Xdg { frame: None, .. } => {}
            ShellSpecificState::WlrLayer { .. } => unreachable!(),
        }
    }

    /// Whether the window is focused.
    #[inline]
    pub fn has_focus(&self) -> bool {
        self.has_focus
    }

    /// Whether the IME is allowed.
    #[inline]
    pub fn ime_allowed(&self) -> bool {
        self.ime_allowed
    }

    /// Get the size of the window.
    #[inline]
    pub fn inner_size(&self) -> LogicalSize<u32> {
        self.size
    }

    /// Whether the window received initial configure event from the compositor.
    #[inline]
    pub fn is_configured(&self) -> bool {
        match &self.shell_specific {
            ShellSpecificState::Xdg { last_configure, .. } => last_configure.is_some(),
            ShellSpecificState::WlrLayer { last_configure, .. } => last_configure.is_some(),
        }
    }

    #[inline]
    pub fn is_decorated(&mut self) -> bool {
        match &self.shell_specific {
            ShellSpecificState::Xdg {
                last_configure,
                frame,
                ..
            } => {
                let csd = last_configure
                    .as_ref()
                    .map(|configure| configure.decoration_mode == DecorationMode::Client)
                    .unwrap_or(false);
                if let Some(frame) = csd.then_some(frame.as_ref()).flatten() {
                    !frame.is_hidden()
                } else {
                    // Server side decorations.
                    true
                }
            }
            ShellSpecificState::WlrLayer { .. } => false,
        }
    }

    /// Create new window state.
    pub fn new_xdg(
        connection: Connection,
        queue_handle: &QueueHandle<WinitState>,
        winit_state: &WinitState,
        initial_size: Size,
        window: Window,
        theme: Option<Theme>,
    ) -> Self {
        let compositor = winit_state.compositor_state.clone();
        let pointer_constraints = winit_state.pointer_constraints.clone();
        let viewport = winit_state
            .viewporter_state
            .as_ref()
            .map(|state| state.get_viewport(window.wl_surface(), queue_handle));
        let fractional_scale = winit_state
            .fractional_scaling_manager
            .as_ref()
            .map(|fsm| fsm.fractional_scaling(window.wl_surface(), queue_handle));

        Self {
            blur: None,
            blur_manager: winit_state.kwin_blur_manager.clone(),
            compositor,
            connection,
            theme,
            cursor_grab_mode: GrabState::new(),
            selected_cursor: Default::default(),
            cursor_visible: true,
            fractional_scale,
            has_focus: false,
            ime_allowed: false,
            ime_purpose: ImePurpose::Normal,
            pointer_constraints,
            pointers: Default::default(),
            queue_handle: queue_handle.clone(),
            scale_factor: 1.,
            shell_specific: ShellSpecificState::Xdg {
                csd_fails: false,
                decorate: true,
                frame: None,
                frame_callback_state: FrameCallbackState::None,
                has_pending_move: None,
                last_configure: None,
                max_inner_size: None,
                min_inner_size: MIN_WINDOW_SIZE,
                resizable: true,
                stateless_size: initial_size.to_logical(1.),
                window,
            },
            shm: winit_state.shm.wl_shm().clone(),
            custom_cursor_pool: winit_state.custom_cursor_pool.clone(),
            size: initial_size.to_logical(1.),
            initial_size: Some(initial_size),
            text_inputs: Vec::new(),
            title: String::default(),
            transparent: false,
            viewport,
        }
    }

    pub fn new_layer(
        connection: Connection,
        queue_handle: &QueueHandle<WinitState>,
        winit_state: &WinitState,
        initial_size: Size,
        layer_surface: LayerSurface,
        theme: Option<Theme>,
    ) -> Self {
        let compositor = winit_state.compositor_state.clone();
        let pointer_constraints = winit_state.pointer_constraints.clone();
        let viewport = winit_state
            .viewporter_state
            .as_ref()
            .map(|state| state.get_viewport(layer_surface.wl_surface(), queue_handle));
        let fractional_scale = winit_state
            .fractional_scaling_manager
            .as_ref()
            .map(|fsm| fsm.fractional_scaling(layer_surface.wl_surface(), queue_handle));

        Self {
            blur: None,
            blur_manager: winit_state.kwin_blur_manager.clone(),
            compositor,
            connection,
            theme,
            cursor_grab_mode: GrabState::new(),
            selected_cursor: Default::default(),
            cursor_visible: true,
            custom_cursor_pool: winit_state.custom_cursor_pool.clone(),
            fractional_scale,
            has_focus: false,
            ime_allowed: false,
            ime_purpose: ImePurpose::Normal,
            pointer_constraints,
            pointers: Default::default(),
            queue_handle: queue_handle.clone(),
            scale_factor: 1.,
            shell_specific: ShellSpecificState::WlrLayer {
                surface: layer_surface,
                last_configure: None,
            },
            shm: winit_state.shm.wl_shm().clone(),

            size: initial_size.to_logical(1.),
            text_inputs: Vec::new(),
            initial_size: Some(initial_size),
            title: String::default(),
            transparent: false,
            viewport,
        }
    }

    /// Get the outer size of the window.
    #[inline]
    pub fn outer_size(&self) -> LogicalSize<u32> {
        match &self.shell_specific {
            ShellSpecificState::Xdg { frame, .. } => frame
                .as_ref()
                .map(|frame| frame.add_borders(self.size.width, self.size.height).into())
                .unwrap_or(self.size),
            ShellSpecificState::WlrLayer { .. } => self.size,
        }
    }

    /// Register pointer on the top-level.
    pub fn pointer_entered(&mut self, added: Weak<ThemedPointer<WinitPointerData>>) {
        self.pointers.push(added);
        self.reload_cursor_style();

        let mode = self.cursor_grab_mode.user_grab_mode;
        let _ = self.set_cursor_grab_inner(mode);
    }

    /// Pointer has left the top-level.
    pub fn pointer_left(&mut self, removed: Weak<ThemedPointer<WinitPointerData>>) {
        let mut new_pointers = Vec::new();
        for pointer in self.pointers.drain(..) {
            if let Some(pointer) = pointer.upgrade() {
                if pointer.pointer() != removed.upgrade().unwrap().pointer() {
                    new_pointers.push(Arc::downgrade(&pointer));
                }
            }
        }

        self.pointers = new_pointers;
    }

    /// Refresh the decorations frame if it's present returning whether the client should redraw.
    pub fn refresh_frame(&mut self) -> bool {
        if let ShellSpecificState::Xdg {frame: Some(ref mut frame), .. } = self.shell_specific {
            if !frame.is_hidden() && frame.is_dirty() {
                return frame.draw();
            }
        }

        false
    }

    /// Reload the cursor style on the given window.
    pub fn reload_cursor_style(&mut self) {
        if self.cursor_visible {
            match &self.selected_cursor {
                SelectedCursor::Named(icon) => self.set_cursor(*icon),
                SelectedCursor::Custom(cursor) => self.apply_custom_cursor(cursor),
            }
        } else {
            self.set_cursor_visible(self.cursor_visible);
        }
    }

    /// Reissue the transparency hint to the compositor.
    pub fn reload_transparency_hint(&self) {
        let surface = self.wl_surface();

        if self.transparent {
            surface.set_opaque_region(None);
        } else if let Ok(region) = Region::new(&*self.compositor) {
            region.add(0, 0, i32::MAX, i32::MAX);
            surface.set_opaque_region(Some(region.wl_region()));
        } else {
            warn!("Failed to mark window opaque.");
        }
    }

    /// Try to resize the window when the user can do so.
    pub fn request_inner_size(&mut self, inner_size: Size) -> PhysicalSize<u32> {
        let scale_factor = self.scale_factor();
        match self.shell_specific {
            ShellSpecificState::Xdg { ref last_configure, .. } => {
                if last_configure
                    .as_ref()
                    .map(Self::is_stateless)
                    .unwrap_or(true) {

                    self.resize(inner_size.to_logical(scale_factor))
                }
            },
            ShellSpecificState::WlrLayer { .. } => {
                self.resize(inner_size.to_logical(scale_factor))
            },
        };

        logical_to_physical_rounded(self.inner_size(), scale_factor)
    }

    /// Resize the window to the new inner size.
    fn resize(&mut self, inner_size: LogicalSize<u32>) {
        self.size = inner_size;

        // Update the stateless size.
        match &mut self.shell_specific {
            ShellSpecificState::Xdg {
                last_configure,
                stateless_size,
                ..
            } => {
                if Some(true) == last_configure.as_ref().map(Self::is_stateless) {
                    *stateless_size = inner_size;
                }
            }
            ShellSpecificState::WlrLayer { .. } => {}
        }

        // Update the inner frame.
        let ((x, y), outer_size) = match self.shell_specific {
            ShellSpecificState::Xdg {
                frame: Some(ref mut frame),
                ..
            } => {
                // Resize only visible frame.
                if !frame.is_hidden() {
                    frame.resize(
                        NonZeroU32::new(self.size.width).unwrap(),
                        NonZeroU32::new(self.size.height).unwrap(),
                    );
                }

                (
                    frame.location(),
                    frame.add_borders(self.size.width, self.size.height).into(),
                )
            }
            _ => ((0, 0), self.size),
        };

        // Reload the hint.
        self.reload_transparency_hint();

        // Set the window geometry.
        match &self.shell_specific {
            ShellSpecificState::Xdg { window, .. } => {
                window.xdg_surface().set_window_geometry(
                    x,
                    y,
                    outer_size.width as i32,
                    outer_size.height as i32,
                );
            }
            ShellSpecificState::WlrLayer { surface, .. } => {
                surface.set_size(outer_size.width, outer_size.height)
            }
        }

        // Update the target viewport, this is used if and only if fractional scaling is in use.
        if let Some(viewport) = self.viewport.as_ref() {
            // Set inner size without the borders.
            viewport.set_destination(self.size.width as _, self.size.height as _);
        }
    }

    /// Get the scale factor of the window.
    #[inline]
    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// Set the cursor icon.
    pub fn set_cursor(&mut self, cursor_icon: CursorIcon) {
        self.selected_cursor = SelectedCursor::Named(cursor_icon);

        if !self.cursor_visible {
            return;
        }

        self.apply_on_poiner(|pointer, _| {
            if pointer.set_cursor(&self.connection, cursor_icon).is_err() {
                warn!("Failed to set cursor to {:?}", cursor_icon);
            }
        })
    }

    /// Set the custom cursor icon.
    pub(crate) fn set_custom_cursor(&mut self, cursor: &CursorImage) {
        let cursor = {
            let mut pool = self.custom_cursor_pool.lock().unwrap();
            CustomCursor::new(&mut pool, cursor)
        };

        if self.cursor_visible {
            self.apply_custom_cursor(&cursor);
        }

        self.selected_cursor = SelectedCursor::Custom(cursor);
    }

    fn apply_custom_cursor(&self, cursor: &CustomCursor) {
        self.apply_on_poiner(|pointer, _| {
            let surface = pointer.surface();

            let scale = surface
                .data::<SurfaceData>()
                .unwrap()
                .surface_data()
                .scale_factor();

            surface.set_buffer_scale(scale);
            surface.attach(Some(cursor.buffer.wl_buffer()), 0, 0);
            if surface.version() >= 4 {
                surface.damage_buffer(0, 0, cursor.w, cursor.h);
            } else {
                surface.damage(0, 0, cursor.w / scale, cursor.h / scale);
            }
            surface.commit();

            let serial = pointer
                .pointer()
                .data::<WinitPointerData>()
                .and_then(|data| data.pointer_data().latest_enter_serial())
                .unwrap();

            pointer.pointer().set_cursor(
                serial,
                Some(surface),
                cursor.hotspot_x / scale,
                cursor.hotspot_y / scale,
            );
        });
    }

    pub fn is_maximized(&self) -> bool {
        match &self.shell_specific {
            ShellSpecificState::Xdg { last_configure, .. } => last_configure
                .as_ref()
                .map(|last_configure| last_configure.is_maximized())
                .unwrap_or_default(),
            ShellSpecificState::WlrLayer { .. } => false,
        }
    }

    pub fn is_fullscreen(&self) -> bool {
        match &self.shell_specific {
            ShellSpecificState::Xdg { last_configure, .. } => last_configure
                .as_ref()
                .map(|last_configure| last_configure.is_fullscreen())
                .unwrap_or_default(),
            ShellSpecificState::WlrLayer { .. } => false,
        }
    }

    /// Set maximum inner window size.
    pub fn set_min_inner_size(&mut self, size: Option<LogicalSize<u32>>) {
        match &mut self.shell_specific {
            ShellSpecificState::Xdg {
                window,
                frame,
                min_inner_size,
                ..
            } => {
                // Ensure that the window has the right minimum size.
                let mut size = size.unwrap_or(MIN_WINDOW_SIZE);
                size.width = size.width.max(MIN_WINDOW_SIZE.width);
                size.height = size.height.max(MIN_WINDOW_SIZE.height);

                // Add the borders.
                let size = frame
                    .as_ref()
                    .map(|frame| frame.add_borders(size.width, size.height).into())
                    .unwrap_or(size);

                *min_inner_size = size;
                window.set_min_size(Some(size.into()));
            }
            ShellSpecificState::WlrLayer { .. } => {
                warn!("Minimum size is ignored for layer_shell windows")
            }
        }
    }

    /// Set maximum inner window size.
    pub fn set_max_inner_size(&mut self, size: Option<LogicalSize<u32>>) {
        match &mut self.shell_specific {
            ShellSpecificState::Xdg {
                window,
                frame,
                max_inner_size,
                ..
            } => {
                let size = size.map(|size| {
                    frame
                        .as_ref()
                        .map(|frame| frame.add_borders(size.width, size.height).into())
                        .unwrap_or(size)
                });

                *max_inner_size = size;
                window.set_max_size(size.map(Into::into));
            }
            ShellSpecificState::WlrLayer { .. } => {
                warn!("Maximum size is ignored for layer_shell windows")
            }
        }
    }

    /// Set the CSD theme.
    pub fn set_theme(&mut self, theme: Option<Theme>) {
        match &mut self.shell_specific {
            ShellSpecificState::Xdg { frame, .. } => {
                self.theme = theme;
                #[cfg(feature = "sctk-adwaita")]
                if let Some(frame) = frame.as_mut() {
                    frame.set_config(into_sctk_adwaita_config(theme))
                }
            }
            ShellSpecificState::WlrLayer { .. } => {
                if theme.is_some() {
                    warn!("Theme is ignored for layer_shell windows")
                }
            }
        }
    }

    /// The current theme for CSD decorations.
    #[inline]
    pub fn theme(&self) -> Option<Theme> {
        match &self.shell_specific {
            ShellSpecificState::Xdg { .. } => self.theme,
            ShellSpecificState::WlrLayer { .. } => None,
        }
    }

    /// Set the cursor grabbing state on the top-level.
    pub fn set_cursor_grab(&mut self, mode: CursorGrabMode) -> Result<(), ExternalError> {
        // Replace the user grabbing mode.
        self.cursor_grab_mode.user_grab_mode = mode;
        self.set_cursor_grab_inner(mode)
    }

    /// Reload the hints for minimum and maximum sizes.
    pub fn reload_min_max_hints(&mut self) {
        match self.shell_specific {
            ShellSpecificState::Xdg {
                min_inner_size,
                max_inner_size,
                ..
            } => {
                self.set_min_inner_size(Some(min_inner_size));
                self.set_max_inner_size(max_inner_size);
            }
            ShellSpecificState::WlrLayer { .. } => {}
        }
    }

    /// Set the grabbing state on the surface.
    fn set_cursor_grab_inner(&mut self, mode: CursorGrabMode) -> Result<(), ExternalError> {
        let pointer_constraints = match self.pointer_constraints.as_ref() {
            Some(pointer_constraints) => pointer_constraints,
            None if mode == CursorGrabMode::None => return Ok(()),
            None => return Err(ExternalError::NotSupported(NotSupportedError::new())),
        };

        // Replace the current mode.
        let old_mode = std::mem::replace(&mut self.cursor_grab_mode.current_grab_mode, mode);

        match old_mode {
            CursorGrabMode::None => (),
            CursorGrabMode::Confined => self.apply_on_poiner(|_, data| {
                data.unconfine_pointer();
            }),
            CursorGrabMode::Locked => {
                self.apply_on_poiner(|_, data| data.unlock_pointer());
            }
        }

        let surface = self.wl_surface();
        match mode {
            CursorGrabMode::Locked => self.apply_on_poiner(|pointer, data| {
                let pointer = pointer.pointer();
                data.lock_pointer(pointer_constraints, surface, pointer, &self.queue_handle)
            }),
            CursorGrabMode::Confined => self.apply_on_poiner(|pointer, data| {
                let pointer = pointer.pointer();
                data.confine_pointer(pointer_constraints, surface, pointer, &self.queue_handle)
            }),
            CursorGrabMode::None => {
                // Current lock/confine was already removed.
            }
        }

        Ok(())
    }

    pub fn show_window_menu(&self, position: LogicalPosition<u32>) {
        match &self.shell_specific {
            ShellSpecificState::Xdg { window, .. } => {
                // TODO(kchibisov) handle touch serials.
                self.apply_on_poiner(|_, data| {
                    let serial = data.latest_button_serial();
                    let seat = data.seat();
                    window.show_window_menu(seat, serial, position.into());
                });
            }
            ShellSpecificState::WlrLayer { .. } => {}
        }
    }

    /// Set the position of the cursor.
    pub fn set_cursor_position(&self, position: LogicalPosition<f64>) -> Result<(), ExternalError> {
        if self.pointer_constraints.is_none() {
            return Err(ExternalError::NotSupported(NotSupportedError::new()));
        }

        // Positon can be set only for locked cursor.
        if self.cursor_grab_mode.current_grab_mode != CursorGrabMode::Locked {
            return Err(ExternalError::Os(os_error!(
                crate::platform_impl::OsError::Misc(
                    "cursor position can be set only for locked cursor."
                )
            )));
        }

        self.apply_on_poiner(|_, data| {
            data.set_locked_cursor_position(position.x, position.y);
        });

        Ok(())
    }

    /// Set the visibility state of the cursor.
    pub fn set_cursor_visible(&mut self, cursor_visible: bool) {
        self.cursor_visible = cursor_visible;

        if self.cursor_visible {
            match &self.selected_cursor {
                SelectedCursor::Named(icon) => self.set_cursor(*icon),
                SelectedCursor::Custom(cursor) => self.apply_custom_cursor(cursor),
            }
        } else {
            for pointer in self.pointers.iter().filter_map(|pointer| pointer.upgrade()) {
                let latest_enter_serial = pointer.pointer().winit_data().latest_enter_serial();

                pointer
                    .pointer()
                    .set_cursor(latest_enter_serial, None, 0, 0);
            }
        }
    }

    /// Whether show or hide client side decorations.
    #[inline]
    pub fn set_decorate(&mut self, decorate: bool) {
        match &mut self.shell_specific {
            ShellSpecificState::Xdg {
                window,
                frame,
                decorate: shell_decorate,
                last_configure,
                ..
            } => {
                if decorate == *shell_decorate {
                    return;
                }

                *shell_decorate = decorate;

                match last_configure
                    .as_ref()
                    .map(|configure| configure.decoration_mode)
                {
                    Some(DecorationMode::Server) if !*shell_decorate => {
                        // To disable decorations we should request client and hide the frame.
                        window
                            .request_decoration_mode(Some(DecorationMode::Client))
                    }
                    _ if *shell_decorate => window
                        .request_decoration_mode(Some(DecorationMode::Server)),
                    _ => (),
                }

                if let Some(frame) = frame.as_mut() {
                    frame.set_hidden(!decorate);
                    // Force the resize.
                    self.resize(self.size);
                }
            }
            ShellSpecificState::WlrLayer { .. } => {
                if decorate {
                    warn!("Client-side decorations are ignored for layer_shell windows");
                }
            }
        }
    }

    /// Mark that the window has focus.
    ///
    /// Should be used from routine that sends focused event.
    #[inline]
    pub fn set_has_focus(&mut self, has_focus: bool) {
        self.has_focus = has_focus;
    }

    /// Returns `true` if the requested state was applied.
    pub fn set_ime_allowed(&mut self, allowed: bool) -> bool {
        self.ime_allowed = allowed;

        let mut applied = false;
        for text_input in &self.text_inputs {
            applied = true;
            if allowed {
                text_input.enable();
                text_input.set_content_type_by_purpose(self.ime_purpose);
            } else {
                text_input.disable();
            }
            text_input.commit();
        }

        applied
    }

    /// Set the IME position.
    pub fn set_ime_cursor_area(&self, position: LogicalPosition<u32>, size: LogicalSize<u32>) {
        // FIXME: This won't fly unless user will have a way to request IME window per seat, since
        // the ime windows will be overlapping, but winit doesn't expose API to specify for
        // which seat we're setting IME position.
        let (x, y) = (position.x as i32, position.y as i32);
        let (width, height) = (size.width as i32, size.height as i32);
        for text_input in self.text_inputs.iter() {
            text_input.set_cursor_rectangle(x, y, width, height);
            text_input.commit();
        }
    }

    /// Set the IME purpose.
    pub fn set_ime_purpose(&mut self, purpose: ImePurpose) {
        self.ime_purpose = purpose;

        for text_input in &self.text_inputs {
            text_input.set_content_type_by_purpose(purpose);
            text_input.commit();
        }
    }

    /// Get the IME purpose.
    pub fn ime_purpose(&self) -> ImePurpose {
        self.ime_purpose
    }

    /// Set the scale factor for the given window.
    #[inline]
    pub fn set_scale_factor(&mut self, scale_factor: f64) {
        self.scale_factor = scale_factor;

        // NOTE: When fractional scaling is not used update the buffer scale.
        if self.fractional_scale.is_none() {
            let _ = self.wl_surface().set_buffer_scale(self.scale_factor as _);
        }

        if let ShellSpecificState::Xdg { frame: Some(ref mut frame), .. } = self.shell_specific {
            frame.set_scaling_factor(scale_factor);
        }
    }

    /// Make window background blurred
    #[inline]
    pub fn set_blur(&mut self, blurred: bool) {
        if blurred && self.blur.is_none() {
            if let Some(blur_manager) = self.blur_manager.as_ref() {
                let blur = blur_manager.blur(self.wl_surface(), &self.queue_handle);
                blur.commit();
                self.blur = Some(blur);
            } else {
                info!("Blur manager unavailable, unable to change blur")
            }
        } else if !blurred && self.blur.is_some() {
            self.blur_manager
                .as_ref()
                .unwrap()
                .unset(self.wl_surface());
            self.blur.take().unwrap().release();
        }
    }

    /// Set the window title to a new value.
    ///
    /// This will autmatically truncate the title to something meaningfull.
    pub fn set_title(&mut self, mut title: String) {
        // Truncate the title to at most 1024 bytes, so that it does not blow up the protocol
        // messages
        if title.len() > 1024 {
            let mut new_len = 1024;
            while !title.is_char_boundary(new_len) {
                new_len -= 1;
            }
            title.truncate(new_len);
        }

        match &mut self.shell_specific {
            ShellSpecificState::Xdg { window, frame, .. } => {
                // Update the CSD title.
                if let Some(frame) = frame.as_mut() {
                    frame.set_title(&title);
                }
                window.set_title(&title);
            }
            ShellSpecificState::WlrLayer { .. } => {}
        }
        self.title = title;
    }

    /// Mark the window as transparent.
    #[inline]
    pub fn set_transparent(&mut self, transparent: bool) {
        self.transparent = transparent;
        self.reload_transparency_hint();
    }

    /// Register text input on the top-level.
    #[inline]
    pub fn text_input_entered(&mut self, text_input: &ZwpTextInputV3) {
        if !self.text_inputs.iter().any(|t| t == text_input) {
            self.text_inputs.push(text_input.clone());
        }
    }

    /// The text input left the top-level.
    #[inline]
    pub fn text_input_left(&mut self, text_input: &ZwpTextInputV3) {
        if let Some(position) = self.text_inputs.iter().position(|t| t == text_input) {
            self.text_inputs.remove(position);
        }
    }

    /// Get the cached title.
    #[inline]
    pub fn title(&self) -> &str {
        &self.title
    }
}

impl Drop for WindowState {
    fn drop(&mut self) {
        if let Some(blur) = self.blur.take() {
            blur.release();
        }

        if let Some(fs) = self.fractional_scale.take() {
            fs.destroy();
        }

        if let Some(viewport) = self.viewport.take() {
            viewport.destroy();
        }

        // NOTE: the wl_surface used by the window is being cleaned up when
        // dropping SCTK `Window`.
    }
}

/// The state of the cursor grabs.
#[derive(Clone, Copy)]
struct GrabState {
    /// The grab mode requested by the user.
    user_grab_mode: CursorGrabMode,

    /// The current grab mode.
    current_grab_mode: CursorGrabMode,
}

impl GrabState {
    fn new() -> Self {
        Self {
            user_grab_mode: CursorGrabMode::None,
            current_grab_mode: CursorGrabMode::None,
        }
    }
}

/// The state of the frame callback.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameCallbackState {
    /// No frame callback was requsted.
    #[default]
    None,
    /// The frame callback was requested, but not yet arrived, the redraw events are throttled.
    Requested,
    /// The callback was marked as done, and user could receive redraw requested
    Received,
}

impl From<ResizeDirection> for XdgResizeEdge {
    fn from(value: ResizeDirection) -> Self {
        match value {
            ResizeDirection::North => XdgResizeEdge::Top,
            ResizeDirection::West => XdgResizeEdge::Left,
            ResizeDirection::NorthWest => XdgResizeEdge::TopLeft,
            ResizeDirection::NorthEast => XdgResizeEdge::TopRight,
            ResizeDirection::East => XdgResizeEdge::Right,
            ResizeDirection::SouthWest => XdgResizeEdge::BottomLeft,
            ResizeDirection::SouthEast => XdgResizeEdge::BottomRight,
            ResizeDirection::South => XdgResizeEdge::Bottom,
        }
    }
}

// NOTE: Rust doesn't allow `From<Option<Theme>>`.
#[cfg(feature = "sctk-adwaita")]
fn into_sctk_adwaita_config(theme: Option<Theme>) -> sctk_adwaita::FrameConfig {
    match theme {
        Some(Theme::Light) => sctk_adwaita::FrameConfig::light(),
        Some(Theme::Dark) => sctk_adwaita::FrameConfig::dark(),
        None => sctk_adwaita::FrameConfig::auto(),
    }
}
