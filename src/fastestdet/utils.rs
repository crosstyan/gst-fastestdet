// we got built-in `tanh` x.tanh()
pub trait Sigmoid {
    fn sigmoid(self) -> Self;
}

impl Sigmoid for f32 {
    fn sigmoid(self) -> f32 {
        1.0 / (1.0 + (-self).exp())
    }
}
