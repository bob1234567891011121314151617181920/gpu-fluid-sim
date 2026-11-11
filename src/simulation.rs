use glam::{UVec2, Vec2, uvec2, vec2};

const GRAVITY: f32 = 9.81;
pub const TIME_STEP: f32 = 1.0 / 144.0;

const CELL_SIZE: f32 = 0.1;
pub const PARTICLE_RADIUS: f32 = CELL_SIZE * 0.15;
const MIN_PARTICLE_DISTANCE: f32 = 2.0 * PARTICLE_RADIUS;
const MIN_PARTICLE_DISTANCE_SQUARED: f32 = MIN_PARTICLE_DISTANCE * MIN_PARTICLE_DISTANCE;
const HASH_CELL_SIZE: f32 = 2.1 * PARTICLE_RADIUS;

const SEPARATION_ITERATIONS: usize = 2;
const PRESSURE_ITERATIONS: usize = 50;
const FLIP_RATIO: f32 = 0.9;
const DRIFT_COMPENSATION_STRENGTH: f32 = 0.6;

const SPAWN_SIZE: UVec2 = uvec2(100, 100);
const SPAWN_SPACING: f32 = 2.0 * PARTICLE_RADIUS;

#[derive(Clone, Copy, PartialEq)]
enum CellType {
    Solid,
    Fluid,
    Air,
}

trait GridIndexing {
    fn dims(&self) -> UVec2;
    fn inv_cell_size(&self) -> f32;

    fn cell_coordinate(&self, position: Vec2) -> UVec2 {
        //as uvec2 convert nagative to 0
        (position * self.inv_cell_size())
            .floor()
            .as_uvec2()
            .min(self.dims() - UVec2::ONE)
    }

    fn cell_index(&self, cell: UVec2) -> usize {
        (cell.x * self.dims().y + cell.y) as usize
    }

    fn cell_index_from_position(&self, position: Vec2) -> usize {
        self.cell_index(self.cell_coordinate(position))
    }
}

struct Particles {
    positions: Vec<Vec2>,
    previous_positions: Vec<Vec2>,
    velocities: Vec<Vec2>,
}

impl Particles {
    fn new(positions: Vec<Vec2>) -> Self {
        let velocities = vec![Vec2::ZERO; positions.len()];
        Self {
            previous_positions: positions.clone(),
            positions,
            velocities,
        }
    }

    fn apply_gravity(&mut self) {
        for velocity in &mut self.velocities {
            velocity.y -= GRAVITY * TIME_STEP;
        }
    }

    fn count(&self) -> usize {
        self.positions.len()
    }
}

struct ParticleHash {
    inv_cell_size: f32,
    dims: UVec2,
    cell_count: usize,
    cell_offsets: Vec<u32>,
    sorted_particle_ids: Vec<u32>,
}

impl GridIndexing for ParticleHash {
    fn dims(&self) -> UVec2 {
        self.dims
    }

    fn inv_cell_size(&self) -> f32 {
        self.inv_cell_size
    }
}

impl ParticleHash {
    fn new(world_size: Vec2, particle_count: usize) -> Self {
        let dims = (world_size / HASH_CELL_SIZE).ceil().as_uvec2();
        let cell_count = (dims.x * dims.y) as usize;
        Self {
            inv_cell_size: 1.0 / HASH_CELL_SIZE,
            dims,
            cell_count,
            cell_offsets: vec![0; cell_count + 1],
            sorted_particle_ids: vec![0; particle_count],
        }
    }

    fn build(&mut self, particles: &Particles) {
        self.cell_offsets.fill(0);

        //count
        for &position in &particles.positions {
            let cell = self.cell_index_from_position(position);
            self.cell_offsets[cell] += 1;
        }

        //prefix sum
        let mut sum = 0;
        for offset in self.cell_offsets[..self.cell_count].iter_mut() {
            sum += *offset;
            *offset = sum;
        }
        self.cell_offsets[self.cell_count] = sum;

        for (id, &position) in particles.positions.iter().enumerate() {
            let cell = self.cell_index_from_position(position);
            self.cell_offsets[cell] -= 1;
            self.sorted_particle_ids[self.cell_offsets[cell] as usize] = id as u32;
        }
    }

    fn particles_in_cell(&self, cell: UVec2) -> &[u32] {
        let index = self.cell_index(cell);
        let start = self.cell_offsets[index] as usize;
        let end = self.cell_offsets[index + 1] as usize;
        &self.sorted_particle_ids[start..end]
    }

    fn push_particles_apart(&mut self, particles: &mut Particles) {
        for _ in 0..SEPARATION_ITERATIONS {
            for i in 0..particles.count() {
                let mut position = particles.positions[i];
                let center = self.cell_coordinate(position);

                let min_x = center.x.saturating_sub(1);
                let min_y = center.y.saturating_sub(1);
                let max_x = (center.x + 1).min(self.dims.x - 1);
                let max_y = (center.y + 1).min(self.dims.y - 1);

                for cx in min_x..=max_x {
                    for cy in min_y..=max_y {
                        for &other in self.particles_in_cell(UVec2::new(cx, cy)) {
                            let other = other as usize;
                            if other == i {
                                continue;
                            }

                            let delta = particles.positions[other] - position;
                            let distance_squared = delta.length_squared();
                            if distance_squared > MIN_PARTICLE_DISTANCE_SQUARED
                                || distance_squared <= f32::EPSILON
                            {
                                continue;
                            }

                            let distance = distance_squared.sqrt();
                            let correction =
                                delta * (0.5 * (MIN_PARTICLE_DISTANCE - distance) / distance);
                            position -= correction;
                            particles.positions[other] += correction;
                        }
                    }
                }
                particles.positions[i] = position;
            }
        }
    }
}

struct MacGrid {
    dims: UVec2,
    cell_count: usize,

    cell_type: Vec<CellType>,
    open_mask: Vec<f32>,

    u_velocity: Vec<f32>,
    v_velocity: Vec<f32>,
    u_weights: Vec<f32>,
    v_weights: Vec<f32>,

    prev_u_velocity: Vec<f32>,
    prev_v_velocity: Vec<f32>,
    u_delta: Vec<f32>,
    v_delta: Vec<f32>,

    density: Vec<f32>,
    rest_density: f32,
}

impl GridIndexing for MacGrid {
    fn dims(&self) -> UVec2 {
        self.dims
    }

    fn inv_cell_size(&self) -> f32 {
        1.0 / CELL_SIZE
    }
}

impl MacGrid {
    fn new(world_size: Vec2) -> Self {
        let dims = (world_size / CELL_SIZE).ceil().as_uvec2();
        assert!(dims.x >= 2 && dims.y >= 2);

        let cell_count = (dims.x * dims.y) as usize;

        let mut open_mask = vec![1.0; cell_count];
        let index = |x: u32, y: u32| (x * dims.y + y) as usize;
        for y in 0..dims.y {
            open_mask[index(0, y)] = 0.0;
            open_mask[index(dims.x - 1, y)] = 0.0;
        }
        for x in 0..dims.x {
            open_mask[index(x, 0)] = 0.0;
            open_mask[index(x, dims.y - 1)] = 0.0;
        }
        let cell_type = open_mask
            .iter()
            .map(|&open| {
                if open == 0.0 {
                    CellType::Solid
                } else {
                    CellType::Air
                }
            })
            .collect();

        Self {
            dims,
            cell_count,
            cell_type,
            open_mask,
            u_velocity: vec![0.0; cell_count],
            v_velocity: vec![0.0; cell_count],
            u_weights: vec![0.0; cell_count],
            v_weights: vec![0.0; cell_count],
            prev_u_velocity: vec![0.0; cell_count],
            prev_v_velocity: vec![0.0; cell_count],
            u_delta: vec![0.0; cell_count],
            v_delta: vec![0.0; cell_count],
            density: vec![0.0; cell_count],
            rest_density: 0.0,
        }
    }

    fn u_face_grid_position(position: Vec2) -> Vec2 {
        (position - vec2(0.0, 0.5 * CELL_SIZE)) / CELL_SIZE
    }

    fn v_face_grid_position(position: Vec2) -> Vec2 {
        (position - vec2(0.5 * CELL_SIZE, 0.0)) / CELL_SIZE
    }

    fn bilinear_stencil(dims: UVec2, grid_position: Vec2) -> [(usize, f32); 4] {
        let max_base = (dims - UVec2::splat(2)).as_vec2();
        let base = grid_position.floor().clamp(Vec2::ZERO, max_base);
        let t = (grid_position - base).clamp(Vec2::ZERO, Vec2::ONE);
        let (x0, y0) = (base.x as u32, base.y as u32);
        let index = |x: u32, y: u32| (x * dims.y + y) as usize;

        [
            (index(x0, y0), (1.0 - t.x) * (1.0 - t.y)),
            (index(x0 + 1, y0), t.x * (1.0 - t.y)),
            (index(x0, y0 + 1), (1.0 - t.x) * t.y),
            (index(x0 + 1, y0 + 1), t.x * t.y),
        ]
    }

    fn splat(
        values: &mut [f32],
        weights: &mut [f32],
        dims: UVec2,
        grid_position: Vec2,
        value: f32,
    ) {
        for (index, weight) in Self::bilinear_stencil(dims, grid_position) {
            values[index] += value * weight;
            weights[index] += weight;
        }
    }

    fn particles_to_grid(&mut self, particles: &Particles) {
        // reset
        self.u_velocity.fill(0.0);
        self.v_velocity.fill(0.0);
        self.u_weights.fill(0.0);
        self.v_weights.fill(0.0);
        for (cell_type, &open) in self.cell_type.iter_mut().zip(&self.open_mask) {
            *cell_type = if open == 0.0 {
                CellType::Solid
            } else {
                CellType::Air
            };
        }

        // mark fluid cells
        for &position in &particles.positions {
            let cell = self.cell_index_from_position(position);
            if self.cell_type[cell] == CellType::Air {
                self.cell_type[cell] = CellType::Fluid;
            }
        }

        // scatter particle velocities to grid
        for (&position, &velocity) in particles.positions.iter().zip(&particles.velocities) {
            Self::splat(
                &mut self.u_velocity,
                &mut self.u_weights,
                self.dims,
                Self::u_face_grid_position(position),
                velocity.x,
            );
            Self::splat(
                &mut self.v_velocity,
                &mut self.v_weights,
                self.dims,
                Self::v_face_grid_position(position),
                velocity.y,
            );
        }

        //normalize
        for (value, &weight) in self.u_velocity.iter_mut().zip(&self.u_weights) {
            if weight > 0.0 {
                *value /= weight;
            }
        }
        for (value, &weight) in self.v_velocity.iter_mut().zip(&self.v_weights) {
            if weight > 0.0 {
                *value /= weight;
            }
        }

        // Faces touching a solid cell cannot carry flow.
        let n = self.dims.y as usize;
        for i in 0..self.dims.x as usize {
            for j in 0..n {
                let cell = i * n + j;
                let is_solid = self.cell_type[cell] == CellType::Solid;
                if is_solid || (i > 0 && self.cell_type[cell - n] == CellType::Solid) {
                    self.u_velocity[cell] = 0.0;
                }
                if is_solid || (j > 0 && self.cell_type[cell - 1] == CellType::Solid) {
                    self.v_velocity[cell] = 0.0;
                }
            }
        }
    }

    fn resolve_wall_collisions(&self, particles: &mut Particles) {
        let min = Vec2::splat(CELL_SIZE + PARTICLE_RADIUS);
        let max = self.dims.as_vec2() * CELL_SIZE - min;

        for (position, velocity) in particles
            .positions
            .iter_mut()
            .zip(&mut particles.velocities)
        {
            if position.x < min.x {
                position.x = min.x;
                velocity.x = 0.0;
            }
            if position.x > max.x {
                position.x = max.x;
                velocity.x = 0.0;
            }
            if position.y < min.y {
                position.y = min.y;
                velocity.y = 0.0;
            }
            if position.y > max.y {
                position.y = max.y;
                velocity.y = 0.0;
            }
        }
    }

    fn update_particle_density(&mut self, particles: &Particles) {
        self.density.fill(0.0);
        for &position in &particles.positions {
            for (index, weight) in Self::bilinear_stencil(
                self.dims,
                (position - Vec2::splat(0.5 * CELL_SIZE)) / CELL_SIZE,
            ) {
                self.density[index] += weight;
            }
        }

        // The first frame defines the "uncompressed" reference density.
        if self.rest_density == 0.0 {
            let mut sum = 0.0;
            let mut fluid_cells = 0;
            for (&density, &cell_type) in self.density.iter().zip(&self.cell_type) {
                if cell_type == CellType::Fluid {
                    sum += density;
                    fluid_cells += 1;
                }
            }
            if fluid_cells > 0 {
                self.rest_density = sum / fluid_cells as f32;
            }
        }
    }

    fn apply_a(&self, input: &[f32], output: &mut [f32]) -> Vec<f32> {
        output.fill(0.0);
        let stride = self.dims.y as usize;
        for x in 1..self.dims.x as usize - 1 {
            for y in 1..self.dims.y as usize - 1 {
                let cell = x * stride + y;
                if self.cell_type[cell] != CellType::Fluid {
                    continue;
                }

                let mut diagonal = 0.0;
                let mut neighbor_sum = 0.0;
                for neighbor in [
                    cell - stride, // left
                    cell + stride, // right
                    cell - 1,      // down
                    cell + 1,      // up
                ] {
                    match self.cell_type[neighbor] {
                        CellType::Fluid => {
                            diagonal += 1.0;
                            neighbor_sum += input[neighbor];
                        }
                        CellType::Air => {
                            diagonal += 1.0;
                        }
                        _ => {}
                    }
                }
                output[cell] = diagonal * input[cell] - neighbor_sum;
            }
        }

        output.to_vec()
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&a, &b)| a * b).sum()
    }

    fn solve_incompressibility(&mut self) {
        self.prev_u_velocity.copy_from_slice(&self.u_velocity);
        self.prev_v_velocity.copy_from_slice(&self.v_velocity);

        let mut ap = vec![0.0; self.cell_count];
        let stride = self.dims.y as usize;
        let mut pressure = vec![0.0; self.cell_count];
        let mut r = vec![0.0; self.cell_count];
        for x in 1..self.dims.x as usize - 1 {
            for y in 1..self.dims.y as usize - 1 {
                let cell = x * stride + y;
                if self.cell_type[cell] != CellType::Fluid {
                    continue;
                }

                let mut divergence = self.u_velocity[cell + stride] - self.u_velocity[cell]
                    + self.v_velocity[cell + 1]
                    - self.v_velocity[cell];

                // Drift compensation: treat over-crowded cells as extra outflow.
                if self.rest_density > 0.0 {
                    let compression = self.density[cell] - self.rest_density;
                    if compression > 0.0 {
                        divergence -= DRIFT_COMPENSATION_STRENGTH * compression;
                    }
                }

                r[cell] = -divergence;
            }
        }
        let mut p = r.clone();
        let mut r_dot_old = Self::dot(&r, &r);

        for _ in 0..PRESSURE_ITERATIONS {
            if r_dot_old < 1e-5 {
                break;
            }
            let ap = self.apply_a(&p, &mut ap);
            let p_dot_ap = Self::dot(&p, &ap);
            let alpha = r_dot_old / p_dot_ap;
            for i in 0..self.cell_count {
                pressure[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }

            let r_dot_new = Self::dot(&r, &r);
            let beta = r_dot_new / r_dot_old;
            for i in 0..self.cell_count {
                p[i] = r[i] + beta * p[i];
            }
            r_dot_old = r_dot_new;
        }

        for x in 1..self.dims.x as usize - 1 {
            for y in 1..self.dims.y as usize - 1 {
                let cell = x * stride + y;
                if self.cell_type[cell] == CellType::Solid {
                    continue;
                }

                let left = cell - stride;
                if self.cell_type[left] != CellType::Solid {
                    self.u_velocity[cell] += pressure[left] - pressure[cell];
                }

                let below = cell - 1;
                if self.cell_type[below] != CellType::Solid {
                    self.v_velocity[cell] += pressure[below] - pressure[cell];
                }
            }
        }

        // store velocity deltas for FLIP
        for i in 0..self.cell_count {
            self.u_delta[i] = self.u_velocity[i] - self.prev_u_velocity[i];
            self.v_delta[i] = self.v_velocity[i] - self.prev_v_velocity[i];
        }
    }

    fn sample(
        &self,
        field: &[f32],
        grid_position: Vec2,
        back_neighbor_stride: usize,
    ) -> Option<f32> {
        let mut weight_sum = 0.0;
        let mut value_sum = 0.0;

        for (index, weight) in Self::bilinear_stencil(self.dims, grid_position) {
            if self.cell_type[index] != CellType::Air
                || (index >= back_neighbor_stride // prevent underflow
                    && self.cell_type[index - back_neighbor_stride] != CellType::Air)
            {
                weight_sum += weight;
                value_sum += field[index] * weight;
            }
        }
        (weight_sum > 0.0).then(|| value_sum / weight_sum)
    }

    fn blend_pic_flip(particle_velocity: f32, pic_velocity: f32, grid_delta: f32) -> f32 {
        let flip_velocity = particle_velocity + grid_delta;
        (1.0 - FLIP_RATIO) * pic_velocity + FLIP_RATIO * flip_velocity
    }

    fn grid_to_particles(&self, particles: &mut Particles) {
        let u_stride = self.dims.y as usize;
        let v_stride = 1;

        for (&position, velocity) in particles.positions.iter().zip(&mut particles.velocities) {
            let u_position = Self::u_face_grid_position(position);
            let v_position = Self::v_face_grid_position(position);

            if let Some(pic) = self.sample(&self.u_velocity, u_position, u_stride) {
                let delta = self
                    .sample(&self.u_delta, u_position, u_stride)
                    .unwrap_or(0.0);
                velocity.x = Self::blend_pic_flip(velocity.x, pic, delta);
            }
            if let Some(pic) = self.sample(&self.v_velocity, v_position, v_stride) {
                let delta = self
                    .sample(&self.v_delta, v_position, v_stride)
                    .unwrap_or(0.0);
                velocity.y = Self::blend_pic_flip(velocity.y, pic, delta);
            }
        }
    }

    fn grid_velocity(&self, position: Vec2) -> Option<Vec2> {
        let u = self.sample(
            &self.u_velocity,
            Self::u_face_grid_position(position),
            self.dims.y as usize,
        )?;
        let v = self.sample(&self.v_velocity, Self::v_face_grid_position(position), 1)?;
        Some(vec2(u, v))
    }

    fn advect_particles(&self, particles: &mut Particles) {
        let dt = TIME_STEP;
        for (position, &velocity) in particles.positions.iter_mut().zip(&particles.velocities) {
            let velocity_at = |p: Vec2| self.grid_velocity(p).unwrap_or(velocity);

            let k1 = velocity_at(*position);
            let k2 = velocity_at(*position + dt * (k1 / 3.0));
            let k3 = velocity_at(*position + dt * (-k1 / 3.0 + k2));
            let k4 = velocity_at(*position + dt * (k1 - k2 + k3));

            *position += dt / 8.0 * (k1 + 3.0 * k2 + 3.0 * k3 + k4);
        }
    }
}

//public api
pub struct FlipFluid {
    particles: Particles,
    particle_hash: ParticleHash,
    mac_grid: MacGrid,
}

impl FlipFluid {
    pub fn new(world_size: Vec2) -> Self {
        let origin = Vec2::splat(CELL_SIZE + PARTICLE_RADIUS);
        let mut positions = Vec::with_capacity((SPAWN_SIZE.x * SPAWN_SIZE.y) as usize);
        for row in 0..SPAWN_SIZE.x {
            for column in 0..SPAWN_SIZE.y {
                let position = origin + vec2(column as f32, row as f32) * SPAWN_SPACING;

                positions.push(position);
            }
        }

        let particle_hash = ParticleHash::new(world_size, positions.len());
        let mac_grid = MacGrid::new(world_size);
        Self {
            particles: Particles::new(positions),
            particle_hash,
            mac_grid,
        }
    }
    pub fn particle_positions(&self, alpha: f32) -> Vec<[f32; 2]> {
        self.particles
            .previous_positions
            .iter()
            .zip(&self.particles.positions)
            .map(|(&previous, &current)| previous.lerp(current, alpha).into())
            .collect()
    }

    pub fn particle_count(&self) -> usize {
        self.particles.count()
    }

    pub fn simulate(&mut self) {
        self.particles
            .previous_positions
            .copy_from_slice(&self.particles.positions);
        self.particles.apply_gravity();

        self.particle_hash.build(&self.particles);
        self.particle_hash.push_particles_apart(&mut self.particles);
        self.mac_grid.particles_to_grid(&self.particles);

        self.mac_grid.update_particle_density(&self.particles);
        self.mac_grid.solve_incompressibility();
        self.mac_grid.grid_to_particles(&mut self.particles);

        self.mac_grid.advect_particles(&mut self.particles);
        self.mac_grid.resolve_wall_collisions(&mut self.particles);
    }
}
