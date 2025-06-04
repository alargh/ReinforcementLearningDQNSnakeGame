use std::io::Write;
use std::process::{Command, Stdio};

pub fn plot_performance(
    gp_stdin: &mut impl Write,
    plot_scores: &[f32],
) {
    write!(
        gp_stdin,
        "plot '-' with lines title 'Scores'\n"
    )
    .unwrap();

    for (i, &score) in plot_scores.iter().enumerate() {
        write!(gp_stdin, "{} {}\n", i + 1, score).unwrap();
    }

    write!(gp_stdin, "e\n").unwrap();

    gp_stdin.flush().unwrap();
}