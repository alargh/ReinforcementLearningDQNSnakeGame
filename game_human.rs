use macroquad::prelude::*;
extern crate rand as external_rand;
use external_rand::{thread_rng, Rng};
use std::collections::VecDeque;

const BLOCK_SIZE: f32 = 20.0;
const SPEED: f32 = 10.0;
const SCREEN_WIDTH: f32 = 640.0;
const SCREEN_HEIGHT: f32 = 480.0;

#[derive(Clone, Copy, PartialEq)]
enum Direction {
    Right,
    Left,
    Up,
    Down,
}

#[derive(Clone, Copy, PartialEq)]
struct Point {
    x: f32,
    y: f32,
}

struct SnakeGame {
    direction: Direction,
    snake: VecDeque<Point>,
    food: Point,
    score: u32,
}

impl SnakeGame {
    fn new() -> Self {
        let head = Point {
            x: SCREEN_WIDTH / 2.0,
            y: SCREEN_HEIGHT / 2.0,
        };
        let mut snake = VecDeque::new();
        snake.push_back(head);
        snake.push_back(Point { x: head.x - BLOCK_SIZE, y: head.y });
        snake.push_back(Point { x: head.x - 2.0 * BLOCK_SIZE, y: head.y });

        let mut game = Self {
            direction: Direction::Right,
            snake,
            food: Point { x: 0.0, y: 0.0 },
            score: 0,
        };
        game.place_food();
        game
    }

    fn place_food(&mut self) {
        let mut rng = thread_rng();
        loop {
            let x = (rng.gen_range(0.0..SCREEN_WIDTH - BLOCK_SIZE) / BLOCK_SIZE).floor() * BLOCK_SIZE;
            let y = (rng.gen_range(0.0..SCREEN_HEIGHT - BLOCK_SIZE) / BLOCK_SIZE).floor() * BLOCK_SIZE;
            let food = Point { x, y };
            if !self.snake.contains(&food) {
                self.food = food;
                break;
            }
        }
    }

    fn update(&mut self) -> bool {
        let mut head = *self.snake.front().unwrap();
        match self.direction {
            Direction::Right => head.x += BLOCK_SIZE,
            Direction::Left => head.x -= BLOCK_SIZE,
            Direction::Down => head.y += BLOCK_SIZE,
            Direction::Up => head.y -= BLOCK_SIZE,
        }

        if self.is_collision(&head) {
            return true;
        }

        self.snake.push_front(head);

        if head == self.food {
            self.score += 1;
            self.place_food();
        } else {
            self.snake.pop_back();
        }

        false
    }

    fn is_collision(&self, head: &Point) -> bool {
        if head.x < 0.0 || head.x >= SCREEN_WIDTH || head.y < 0.0 || head.y >= SCREEN_HEIGHT {
            return true;
        }
        self.snake.iter().skip(1).any(|&p| p == *head)
    }

    fn change_direction(&mut self, new_direction: Direction) {
        let opposite = match self.direction {
            Direction::Right => Direction::Left,
            Direction::Left => Direction::Right,
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
        };
        if new_direction != opposite {
            self.direction = new_direction;
        }
    }

    fn draw(&self) {
        clear_background(BLACK);

        for p in &self.snake {
            draw_rectangle(p.x, p.y, BLOCK_SIZE, BLOCK_SIZE, BLUE);
            draw_rectangle(p.x + 4.0, p.y + 4.0, 12.0, 12.0, DARKBLUE);
        }

        draw_rectangle(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE, RED);

        draw_text(&format!("Score: {}", self.score), 10.0, 30.0, 30.0, WHITE);
    }
}

#[macroquad::main("Snake Game")]
async fn main() {
    let mut game = SnakeGame::new();
    let mut frame_timer = 0.0;

    loop {
        if is_key_pressed(KeyCode::Up) {
            game.change_direction(Direction::Up);
        } else if is_key_pressed(KeyCode::Down) {
            game.change_direction(Direction::Down);
        } else if is_key_pressed(KeyCode::Left) {
            game.change_direction(Direction::Left);
        } else if is_key_pressed(KeyCode::Right) {
            game.change_direction(Direction::Right);
        }

        frame_timer += get_frame_time() * SPEED;
        if frame_timer >= 1.0 {
            frame_timer = 0.0;
            if game.update() {
                println!("Final Score: {}", game.score);
                break;
            }
        }

        game.draw();
        next_frame().await;
    }
}
