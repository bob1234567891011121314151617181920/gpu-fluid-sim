use crate::simulation::{self, FlipFluid};
use glam::Vec2;
use wgpu::util::DeviceExt;

const WORLD_SIZE: Vec2 = Vec2::splat(10.0);

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    size: [f32; 2],
    radius: f32,
}
pub struct SimulationHandler {
    pub instance_buffer: wgpu::Buffer,
    pub globals: Globals,
    pub flip_fluid: simulation::FlipFluid,
    accumulator: f32,
    pub pasued: bool,
}

impl SimulationHandler {
    pub fn new(device: &wgpu::Device) -> Self {
        // will fix later, for now just use a fixed number of particles
        let max_particles = 10000;
        let particle_data = vec![[0.0_f32; 2]; max_particles];
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Instance Buffer"),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            contents: bytemuck::cast_slice(&particle_data),
        });

        let globals = Globals {
            size: WORLD_SIZE.into(),
            radius: simulation::PARTICLE_RADIUS,
        };

        let flip_fluid = FlipFluid::new(WORLD_SIZE);
        Self {
            instance_buffer,
            globals,
            flip_fluid,
            accumulator: 0.0,
            pasued: true,
        }
    }

    pub fn update(&mut self, queue: &wgpu::Queue, dt: f32) {
        if self.pasued {
            return;
        }
        self.accumulator += dt;
        while self.accumulator >= simulation::TIME_STEP {
            self.flip_fluid.simulate();
            self.accumulator -= simulation::TIME_STEP;
        }
        let alpha = self.accumulator / simulation::TIME_STEP;
        queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.flip_fluid.particle_positions(alpha)),
        );
    }
}
