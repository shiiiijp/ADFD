class EarlyStopping:
    def __init__(self, patience=5, delta=0.0001, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose      # whether to show the progress
        self.counter = 0            # current count
        self.best_score = None      # best score
        self.early_stop = False     # flag for early stopping

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:  # 1st epoch
            self.best_score = score
        elif abs(score - self.best_score) < self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0  # reset counter
