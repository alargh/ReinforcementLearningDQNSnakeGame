use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use std::collections::VecDeque;
use ::rand::thread_rng;
use ::rand::Rng;

use crate::model::{LinearQNet, QTrainer};
use crate::game::{Direction, Point, SnakeGame};
use crate::bfs::is_reachable;

const BLOCK_SIZE: f32 = 20.0;
const MAX_MEMORY: usize = 100_000;
const BATCH_SIZE: usize = 1000;
const LR: f64 = 0.001;

// Agent
pub struct Agent {
    pub n_games: u32,
    pub epsilon: i32,
    pub memory: VecDeque<(Tensor, Tensor, Tensor, Tensor, Tensor)>, // (state, action, reward, next_state, done)
    pub trainer: QTrainer,
}

impl Agent {
    pub fn new() -> Self {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = LinearQNet::new(&vs.root(), 14, 256, 3);
        let trainer = QTrainer::new(&vs, model, LR, 0.9);

        Self {
            n_games: 0,
            epsilon: 80,
            memory: VecDeque::with_capacity(MAX_MEMORY),
            trainer,
        }
    }

    pub fn get_state(&self, game: &SnakeGame) -> Tensor {
        let head = game.snake[0];
        let dir_r = game.direction == Direction::Right;
        let dir_l = game.direction == Direction::Left;
        let dir_u = game.direction == Direction::Up;
        let dir_d = game.direction == Direction::Down;

        let point_l = Point { x: head.x - BLOCK_SIZE, y: head.y };
        let point_r = Point { x: head.x + BLOCK_SIZE, y: head.y };
        let point_u = Point { x: head.x, y: head.y - BLOCK_SIZE };
        let point_d = Point { x: head.x, y: head.y + BLOCK_SIZE };

        let danger_straight = (dir_r && game.is_collision_point(point_r))
            || (dir_l && game.is_collision_point(point_l))
            || (dir_u && game.is_collision_point(point_u))
            || (dir_d && game.is_collision_point(point_d));

        let danger_right = (dir_u && game.is_collision_point(point_r))
            || (dir_d && game.is_collision_point(point_l))
            || (dir_l && game.is_collision_point(point_u))
            || (dir_r && game.is_collision_point(point_d));

        let danger_left = (dir_d && game.is_collision_point(point_r))
            || (dir_u && game.is_collision_point(point_l))
            || (dir_r && game.is_collision_point(point_u))
            || (dir_l && game.is_collision_point(point_d));

        let dir_l_flag = dir_l as i64;
        let dir_r_flag = dir_r as i64;
        let dir_u_flag = dir_u as i64;
        let dir_d_flag = dir_d as i64;

        let food_left  = (game.food.x < head.x) as i64;
        let food_right = (game.food.x > head.x) as i64;
        let food_up    = (game.food.y < head.y) as i64;
        let food_down  = (game.food.y > head.y) as i64;

        let clock_wise = [
            Direction::Right,
            Direction::Down,
            Direction::Left,
            Direction::Up,
        ];
        let idx = clock_wise.iter().position(|&d| d == game.direction).unwrap();

        let dir_if_straight = game.direction;
        let dir_if_right = clock_wise[(idx + 1) % 4];
        let dir_if_left = clock_wise[(idx + 3) % 4];

        let head_if_straight = {
            let mut p = head;
            match dir_if_straight {
                Direction::Right => p.x += BLOCK_SIZE,
                Direction::Left  => p.x -= BLOCK_SIZE,
                Direction::Down  => p.y += BLOCK_SIZE,
                Direction::Up    => p.y -= BLOCK_SIZE,
            }
            p
        };
        let head_if_right = {
            let mut p = head;
            match dir_if_right {
                Direction::Right => p.x += BLOCK_SIZE,
                Direction::Left  => p.x -= BLOCK_SIZE,
                Direction::Down  => p.y += BLOCK_SIZE,
                Direction::Up    => p.y -= BLOCK_SIZE,
            }
            p
        };
        let head_if_left = {
            let mut p = head;
            match dir_if_left {
                Direction::Right => p.x += BLOCK_SIZE,
                Direction::Left  => p.x -= BLOCK_SIZE,
                Direction::Down  => p.y += BLOCK_SIZE,
                Direction::Up    => p.y -= BLOCK_SIZE,
            }
            p
        };

        let reachable_straight = is_reachable(head_if_straight, game.food, &game.snake) as i64;
        let reachable_right    = is_reachable(head_if_right,    game.food, &game.snake) as i64;
        let reachable_left     = is_reachable(head_if_left,     game.food, &game.snake) as i64;

        let state_array = [
            danger_straight  as i64,
            danger_right     as i64,
            danger_left      as i64,
            dir_l_flag,
            dir_r_flag,
            dir_u_flag,
            dir_d_flag,
            food_left,
            food_right,
            food_up,
            food_down,
            reachable_straight,
            reachable_right,
            reachable_left,
        ];

        Tensor::from_slice(&state_array)
            .view([1, 14])
            .to_kind(Kind::Float)
    }

    pub fn remember(
        &mut self,
        state: Tensor,     // shape [1, 11], Float
        action: Tensor,    // shape [1, 3], Long (one-hot)
        reward: Tensor,    // shape [1], Float
        next_state: Tensor, // shape [1, 11], Float
        done: Tensor,      // shape [1], Float
    ) {
        if self.memory.len() == MAX_MEMORY {
            self.memory.pop_front();
        }
        self.memory.push_back((state, action, reward, next_state, done));
    }

    pub fn train_long_memory(&mut self) {
        if self.memory.len() < BATCH_SIZE {
            return;
        }

        let mut states = Vec::with_capacity(BATCH_SIZE);
        let mut actions = Vec::with_capacity(BATCH_SIZE);
        let mut rewards = Vec::with_capacity(BATCH_SIZE);
        let mut next_states = Vec::with_capacity(BATCH_SIZE);
        let mut dones = Vec::with_capacity(BATCH_SIZE);

        for (s, a, r, ns, d) in self.memory.iter().take(BATCH_SIZE) {
            states.push(s.copy());
            actions.push(a.copy());
            rewards.push(r.copy());
            next_states.push(ns.copy());
            dones.push(d.copy());
        }

        let states = Tensor::cat(&states, 0);         // shape [BATCH_SIZE, 11], Float
        let actions = Tensor::cat(&actions, 0);       // shape [BATCH_SIZE, 3],  Long
        let rewards = Tensor::cat(&rewards, 0);       // shape [BATCH_SIZE],     Float
        let next_states = Tensor::cat(&next_states, 0); // shape [BATCH_SIZE, 11], Float
        let dones = Tensor::cat(&dones, 0);           // shape [BATCH_SIZE],     Float

        self.trainer
            .train_step(&states, &actions, &rewards, &next_states, &dones);
    }

    pub fn train_short_memory(
        &mut self,
        state: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        next_state: &Tensor,
        done: &Tensor,
    ) {
        self.trainer
            .train_step(state, action, reward, next_state, done);
    }

    pub fn get_action(&mut self, state: &Tensor) -> Tensor {
        let mut rng = thread_rng();
        let mut action = [0, 0, 0];

        if rng.gen_range(0..200) < self.epsilon {
            let move_idx = rng.gen_range(0..3);
            action[move_idx] = 1;
        } else {
            let prediction = self.trainer.model.forward(state); // shape [1, 3]
            let move_idx = prediction.argmax(1, false).int64_value(&[0]) as usize;
            action[move_idx] = 1;
        }

        Tensor::from_slice(&action).view([1, 3])
    }
}