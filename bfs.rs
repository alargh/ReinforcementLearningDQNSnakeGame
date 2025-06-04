use std::{
    collections::{HashSet, VecDeque},
};
use crate::game::Point;

const BLOCK_SIZE: f32 = 20.0;
const SCREEN_WIDTH: f32 = 640.0;
const SCREEN_HEIGHT: f32 = 480.0;

pub fn is_reachable(start: Point, food: Point, snake: &std::collections::VecDeque<Point>) -> bool {
    let grid_w = (SCREEN_WIDTH / BLOCK_SIZE) as i32;
    let grid_h = (SCREEN_HEIGHT / BLOCK_SIZE) as i32;

    let to_cell = |p: Point| {
        ((p.x / BLOCK_SIZE) as i32, (p.y / BLOCK_SIZE) as i32)
    };

    let start_cell = to_cell(start);
    let food_cell  = to_cell(food);

    if start_cell.0 < 0
        || start_cell.1 < 0
        || start_cell.0 >= grid_w
        || start_cell.1 >= grid_h
    {
        return false;
    }

    let mut blocked = HashSet::new();
    for &seg in snake.iter() {
        let c = to_cell(seg);
        blocked.insert(c);
    }

    if blocked.contains(&start_cell) {
        return false;
    }

    let mut visited = vec![vec![false; grid_h as usize]; grid_w as usize];
    let mut queue = VecDeque::new();

    visited[start_cell.0 as usize][start_cell.1 as usize] = true;
    queue.push_back(start_cell);

    let neighbours = [ (1, 0), (-1, 0), (0, 1), (0, -1) ];

    while let Some((x, y)) = queue.pop_front() {
        if (x, y) == food_cell {
            return true;
        }

        for (dx, dy) in neighbours.iter() {
            let nx = x + dx;
            let ny = y + dy;

            if nx < 0 || ny < 0 || nx >= grid_w || ny >= grid_h {
                continue;
            }

            if visited[nx as usize][ny as usize] {
                continue;
            }

            if blocked.contains(&(nx, ny)) {
                continue;
            }

            visited[nx as usize][ny as usize] = true;
            queue.push_back((nx, ny));
        }
    }

    false
}
