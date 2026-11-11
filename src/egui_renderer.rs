use egui::Context;
use egui_wgpu::wgpu::{CommandEncoder, Device, Queue, StoreOp, TextureFormat, TextureView};
use egui_wgpu::{Renderer, ScreenDescriptor, wgpu};
use egui_winit::State;
use winit::window::Window;

pub struct EguiRenderer {
    state: State,
    renderer: Renderer,
}

impl EguiRenderer {
    pub fn new(device: &Device, window: &Window, output_color_format: TextureFormat) -> Self {
        let state = egui_winit::State::new(
            Context::default(),
            egui::ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let renderer = Renderer::new(
            device,
            output_color_format,
            egui_wgpu::RendererOptions::default(),
        );
        Self { state, renderer }
    }

    pub fn get_egui_context(&self) -> &Context {
        self.state.egui_ctx()
    }

    pub fn draw(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        screen_descriptor: &ScreenDescriptor,
        window: &Window,
        window_surface_view: &TextureView,
        ui: impl FnOnce(&Context),
    ) {
        let raw_input = self.state.take_egui_input(window);
        self.get_egui_context().begin_pass(raw_input);

        ui(self.get_egui_context());

        self.get_egui_context()
            .set_pixels_per_point(screen_descriptor.pixels_per_point);
        let mut full_output = self.get_egui_context().end_pass();
        self.state
            .handle_platform_output(window, full_output.platform_output);
        for (id, image_deltas) in &full_output.textures_delta.set {
            for image_delta in image_deltas {
                self.renderer
                    .update_texture(device, queue, *id, image_delta);
            }
        }
        let tris = self.state.egui_ctx().tessellate(
            full_output.shapes,
            self.get_egui_context().pixels_per_point(),
        );
        self.renderer
            .update_buffers(device, queue, encoder, &tris, screen_descriptor);
        let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: window_surface_view,
                resolve_target: None,
                ops: egui_wgpu::wgpu::Operations {
                    load: egui_wgpu::wgpu::LoadOp::Load,
                    store: StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            label: Some("egui main render pass"),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        self.renderer
            .render(&mut render_pass.forget_lifetime(), &tris, screen_descriptor);
        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
        full_output.textures_delta.clear();
    }
}
