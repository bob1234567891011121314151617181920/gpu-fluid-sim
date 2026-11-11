use winit::window::Window;

use crate::egui_renderer::EguiRenderer;

const FPS_SMOOTHING_SECONDS: f32 = 0.5;

struct UiData {
    smoothed_frame_time: f32,
    fps: f32,
}

impl Default for UiData {
    fn default() -> Self {
        Self {
            smoothed_frame_time: 0.0,
            fps: 0.0,
        }
    }
}

impl UiData {
    fn record_frame(&mut self, frame_time: f32) {
        // weight by frame time so the smoothing speed doesn't depend on the frame rate
        let alpha = 1.0 - (-frame_time / FPS_SMOOTHING_SECONDS).exp();
        self.smoothed_frame_time += alpha * (frame_time - self.smoothed_frame_time);
        self.fps = 1.0 / self.smoothed_frame_time;
    }
}

pub struct UiHandler {
    renderer: EguiRenderer,
    ui_data: UiData,
}

impl UiHandler {
    pub fn new(
        device: &wgpu::Device,
        window: &Window,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            renderer: EguiRenderer::new(device, window, surface_format),
            ui_data: UiData::default(),
        }
    }

    pub fn render_ui(
        &mut self,
        window: &Window,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        output: &wgpu::SurfaceTexture,
        config: &wgpu::SurfaceConfiguration,
        width: u32,
        height: u32,
    ) {
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(config.view_formats[0]),
            ..Default::default()
        });

        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: window.scale_factor() as f32,
        };

        let fps = self.ui_data.fps;

        self.renderer.draw(
            device,
            queue,
            encoder,
            &screen_descriptor,
            window,
            &view,
            |ctx| {
                egui::Window::new("debug").show(ctx, |ui| {
                    ui.label(format!("FPS: {fps:.1}"));
                });
            },
        );
    }

    pub fn update(&mut self, dt: f32) {
        self.ui_data.record_frame(dt);
    }
}
