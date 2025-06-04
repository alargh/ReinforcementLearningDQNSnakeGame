use macroquad::prelude::*;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use std::collections::VecDeque;
use std::io::Write;
use std::process::{Command, Stdio};
use ::rand::thread_rng;
use ::rand::Rng;

const BLOCK_SIZE: f32 = 20.0;
const SCREEN_WIDTH: f32 = 640.0;
const SCREEN_HEIGHT: f32 = 480.0;


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Right,
    Left,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}




pub struct SnakeGame {
    pub direction: Direction,
    pub snake: VecDeque<Point>,
    pub food: Point,
    pub score: u32,
    pub frame_iteration: u32,
}


impl SnakeGame {
    pub fn new() -> Self {
        let head = Point {
            x: SCREEN_WIDTH / 2.0,
            y: SCREEN_HEIGHT / 2.0,
        };
        let mut snake = VecDeque::new();
        snake.push_back(head);
        snake.push_back(Point {
            x: head.x - BLOCK_SIZE,
            y: head.y,
        });
        snake.push_back(Point {
            x: head.x - 2.0 * BLOCK_SIZE,
            y: head.y,
        });

        let mut game = Self {
            direction: Direction::Right,
            snake,
            food: Point { x: 0.0, y: 0.0 },
            score: 0,
            frame_iteration: 0,
        };
        game.place_food();
        game
    }

    pub fn reset(&mut self) {
        let head = Point {
            x: SCREEN_WIDTH / 2.0,
            y: SCREEN_HEIGHT / 2.0,
        };
        self.snake.clear();
        self.snake.push_back(head);
        self.snake.push_back(Point {
            x: head.x - BLOCK_SIZE,
            y: head.y,
        });
        self.snake.push_back(Point {
            x: head.x - 2.0 * BLOCK_SIZE,
            y: head.y,
        });
        self.direction = Direction::Right;
        self.score = 0;
        self.frame_iteration = 0;
        self.place_food();
    }

    pub fn place_food(&mut self) {
        let mut rng = thread_rng();
        loop {
            let x = (rng.gen_range(0..(SCREEN_WIDTH / BLOCK_SIZE) as u32) as f32) * BLOCK_SIZE;
            let y = (rng.gen_range(0..(SCREEN_HEIGHT / BLOCK_SIZE) as u32) as f32) * BLOCK_SIZE;
            let food = Point { x, y };

            if !self.snake.iter().any(|&p| p == food) {
                self.food = food;
                break;
            }
        }
    }

    pub fn play_step(&mut self, action: &Tensor) -> (f64, bool, u32) {
        self.frame_iteration += 1;

        let action_arr = [
            action.int64_value(&[0, 0]),
            action.int64_value(&[0, 1]),
            action.int64_value(&[0, 2]),
        ];

        let clock_wise = [
            Direction::Right,
            Direction::Down,
            Direction::Left,
            Direction::Up,
        ];
        let idx = clock_wise.iter().position(|&d| d == self.direction).unwrap();

        let new_dir = if action_arr[2] == 1 {
            clock_wise[(idx + 3) % 4]
        } else if action_arr[1] == 1 {
            clock_wise[(idx + 1) % 4]
        } else {
            self.direction
        };
        self.direction = new_dir;

        let mut head = *self.snake.front().unwrap();
        match self.direction {
            Direction::Right => head.x += BLOCK_SIZE,
            Direction::Left => head.x -= BLOCK_SIZE,
            Direction::Down => head.y += BLOCK_SIZE,
            Direction::Up => head.y -= BLOCK_SIZE,
        }
        self.snake.push_front(head);

        let mut reward = 0.0;
        let mut game_over = false;
        if self.is_collision() || self.frame_iteration > 50 * self.snake.len() as u32 {
            game_over = true;
            reward = -10.0;
            return (reward, game_over, self.score);
        }

        if head == self.food {
            self.score += 1;
            reward = 10.0;
            self.place_food();
        } else {
            self.snake.pop_back();
        }

        (reward, game_over, self.score)
    }

    pub fn is_collision(&self) -> bool {

        let head = *self.snake.front().unwrap();
        if head.x < 0.0 || head.x >= SCREEN_WIDTH || head.y < 0.0 || head.y >= SCREEN_HEIGHT {
            return true;
        }
        self.snake.iter().skip(1).any(|&segment| segment == head)
    }

    pub fn is_collision_point(&self, pt: Point) -> bool {
        if pt.x < 0.0 || pt.x >= SCREEN_WIDTH || pt.y < 0.0 || pt.y >= SCREEN_HEIGHT {
            return true;
        }
        self.snake.iter().any(|&segment| segment == pt)
    }

    pub fn draw(&self) {
        for (i, &point) in self.snake.iter().enumerate() {
            draw_rectangle(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE, BLUE);
            draw_rectangle(point.x + 4.0, point.y + 4.0, 12.0, 12.0, DARKBLUE);

        }

        draw_rectangle(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE, RED);

        draw_text(&format!("Score: {}", self.score), 10.0, 25.0, 25.0, WHITE);
    }
}