use macroquad::prelude::*;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use std::collections::VecDeque;
use std::io::Write;
use std::process::{Command, Stdio};


mod model;
mod agent;
mod game;
mod plot;
mod bfs;

use model::{LinearQNet, QTrainer};
use agent::Agent;
use game::SnakeGame;
use plot::plot_performance;


const SPEED: f32 = 40.0;


#[macroquad::main("Snake")]
async fn main() {
    let mut gnuplot_child = Command::new("gnuplot")
        .arg("-persist")
        .stdin(Stdio::piped())
        .spawn()
        .expect("Failed to start gnuplot");
    let mut gp_stdin = gnuplot_child
        .stdin
        .take()
        .expect("Failed to open stdin for gnuplot");

    let mut game = SnakeGame::new();
    let mut agent = Agent::new();
    let mut plot_scores = Vec::new();
    let mut total_score = 0;
    let mut record = 0;
    let mut timer = 0.0;

    loop {
        let state_old = agent.get_state(&game);

        let action = agent.get_action(&state_old);

        let (reward, done, score) = game.play_step(&action);

        let state_new = agent.get_state(&game);

        let reward_tensor = Tensor::from(reward).to_kind(Kind::Float).view([1]);
        let done_tensor = Tensor::from(done as i64).to_kind(Kind::Float).view([1]);

        agent.train_short_memory(
            &state_old,
            &action,
            &reward_tensor,
            &state_new,
            &done_tensor,
        );

        agent.remember(
            state_old,
            action,
            reward_tensor.copy(),
            state_new,
            done_tensor.copy(),
        );

        if done {
            game.reset();
            agent.n_games += 1;
            agent.epsilon = 80 - agent.n_games.min(80) as i32;
            agent.train_long_memory();

            if score > record {
                record = score;
            }

            plot_scores.push(score as f32);
            total_score += score;

            println!(
                "Game: {}, Score: {}, Record: {}",
                agent.n_games, score, record
            );

            plot_performance(&mut gp_stdin, &plot_scores);
        }

        clear_background(BLACK);
        game.draw();

        timer += get_frame_time();
        if timer < 1.0 / SPEED {
            next_frame().await;
            continue;
        }
        timer = 0.0;

        next_frame().await;
    }
}
