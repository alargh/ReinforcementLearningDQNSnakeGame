use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

#[derive(Debug)]
pub struct LinearQNet {
    linear1: nn::Linear,
    linear2: nn::Linear,
}

impl LinearQNet {
    pub fn new(vs: &nn::Path, input_size: i64, hidden_size: i64, output_size: i64) -> Self {
        let linear1 = nn::linear(vs, input_size, hidden_size, Default::default());
        let linear2 = nn::linear(vs, hidden_size, output_size, Default::default());
        Self { linear1, linear2 }
    }
}

impl Module for LinearQNet {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.linear1).relu().apply(&self.linear2)
    }
}

pub struct QTrainer {
    pub model: LinearQNet,
    optimizer: nn::Optimizer,
    gamma: f64,
}

impl QTrainer {
    pub fn new(vs: &nn::VarStore, model: LinearQNet, lr: f64, gamma: f64) -> Self {
        let optimizer = nn::Adam::default().build(vs, lr).unwrap();
        Self { model, optimizer, gamma }
    }

    pub fn train_step(
        &mut self,
        state: &Tensor,       // Float tensor of shape [batch_size, 11]
        action: &Tensor,      // Long tensor of shape [batch_size, 3]
        reward: &Tensor,      // Float tensor of shape [batch_size]
        next_state: &Tensor,  // Float tensor of shape [batch_size, 11]
        done: &Tensor,        // Float tensor of shape [batch_size]
    ) {
        self.optimizer.zero_grad();

        let pred = self.model.forward(state);

        let mut target = pred.copy();

        let next_q = self.model.forward(next_state);
        let max_next_q = next_q.max_dim(1, false).0;

        let one = Tensor::from(1.0).to_kind(done.kind());
        let not_done = &one - done;

        // q_new = reward + (1 - done) * gamma * max_next_q
        let q_new = reward + not_done * self.gamma * &max_next_q;


        let action_indices = action.argmax(1, false).unsqueeze(-1);
        let gathered = target.gather(1, &action_indices, false);
        target = target.scatter_add(1, &action_indices, &(q_new.unsqueeze(1) - gathered));

        let loss = target.mse_loss(&pred, tch::Reduction::Mean);

        loss.backward();
        self.optimizer.step();
    }
}