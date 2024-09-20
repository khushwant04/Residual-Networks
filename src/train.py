import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from src.helper import accuracy_fn

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler=None,
        gradient_clip_value=None,  # Optional: for gradient clipping
        log_dir='runs/experiment'  # Set default TensorBoard log directory
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler  # Optional: learning rate scheduler
        self.device = device
        self.gradient_clip_value = gradient_clip_value  # Optional: gradient clipping
        self.writer = SummaryWriter(log_dir)  # TensorBoard writer with log directory

        self.logger = logging.getLogger(__name__)

    def _train_one_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()  # Set model to training mode
        train_loss, train_acc = 0.0, 0.0

        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Training Epoch {epoch}')
        for batch_idx, (X, y) in loop:
            X, y = X.to(self.device), y.to(self.device)

            # Forward pass
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Optional gradient clipping for stability
            if self.gradient_clip_value:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)

            self.optimizer.step()

            # Calculate metrics
            train_loss += loss.item()
            train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

            # Update tqdm description
            loop.set_postfix(loss=loss.item(), acc=train_acc / (batch_idx + 1))

        # Log averaged metrics to TensorBoard
        avg_train_loss = train_loss / len(self.train_loader)
        avg_train_acc = train_acc / len(self.train_loader)
        self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
        self.writer.add_scalar('Accuracy/train', avg_train_acc, epoch)

        self.logger.info(f'Train Epoch {epoch}: Loss: {avg_train_loss:.5f} | Accuracy: {avg_train_acc:.2f}')

    def _test_one_epoch(self, epoch):
        """Evaluate the model after one epoch."""
        self.model.eval()  # Set model to evaluation mode
        test_loss, test_acc = 0.0, 0.0  # Initialize loss and accuracy

        with torch.no_grad():
            loop = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f'Testing Epoch {epoch}')
            for batch_idx, (X, y) in loop:
                X, y = X.to(self.device), y.to(self.device)

                # Forward pass
                test_pred = self.model(X)
                loss = self.loss_fn(test_pred, y)

                # Calculate metrics
                test_loss += loss.item()
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

                # Update tqdm description
                loop.set_postfix(loss=loss.item(), acc=test_acc / (batch_idx + 1))

        # Calculate average loss and accuracy over the test set
        self.avg_test_loss = test_loss / len(self.test_loader)  # Store as instance variable
        avg_test_acc = test_acc / len(self.test_loader)

        # Log averaged metrics to TensorBoard
        self.writer.add_scalar('Loss/test', self.avg_test_loss, epoch)
        self.writer.add_scalar('Accuracy/test', avg_test_acc, epoch)

        print(f'Test Epoch {epoch}: Loss: {self.avg_test_loss:.5f} | Accuracy: {avg_test_acc:.2f}')

    def train(self, num_epochs):
        """Train and evaluate the model for a number of epochs."""
        for epoch in range(1, num_epochs + 1):
            self._train_one_epoch(epoch)
            self._test_one_epoch(epoch)

            # Step the scheduler if provided
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # For schedulers like ReduceLROnPlateau that require a metric
                    self.scheduler.step(self.avg_test_loss)
                else:
                    self.scheduler.step()

        # Close the TensorBoard writer
        self.writer.close()
