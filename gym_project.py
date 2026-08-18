import numpy as np
class default:
    def __init__(self):
        self.mean =None
        self.std =None

    def transform(self, X):
        self.mean =np.mean(X, axis=0)
        self.std =np.std(X, axis=0)
        self.std[self.std == 0] = 1
        return (X-self.mean) / self.std

    def transformed(self, X):
        return (X - self.mean)/self.std


def MSE(real_value, guess_value):
        return np.mean((real_value -guess_value) ** 2)

def RMSE(real_value, guess_value):
        return np.sqrt(MSE(real_value,guess_value))

def mae(real_value, guess_value):
        return np.mean(np.abs(real_value -guess_value))

def mape(real_value, guess_value):
        return np.mean(np.abs((real_value -guess_value) / real_value)) *100

def r2_score(real_value, guess_value):
        ss_res = np.sum((real_value -guess_value) ** 2)
        ss_tot = np.sum((real_value -np.mean(real_value)) ** 2)
        return 1 - (ss_res /ss_tot)
def print_all_metrics(real_value, guess_value):
    print("model resuts on test data")
    print("mse -",MSE(real_value,guess_value))
    print("rmse-",RMSE(real_value,guess_value))
    print("mae -",mae(real_value,guess_value))
    print("mape-",mape(real_value,guess_value), "%")
    print("r square(r2)  -",r2_score(real_value,guess_value))
class LinearRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape(-1, 1)

        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

        for i in range(self.iterations):
            guess_value= X @ self.weights + self.bias
            error = y - guess_value

            grad_w = -(2 / n_samples) * (X.T @ error)
            grad_b = -(2 / n_samples) * np.sum(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            if i % 200 == 0:
                print(f"iterations {i}, Loss: {loss}")    
    def predict(self, X):
            return (X @ self.weights + self.bias).flatten()


if __name__ == "__main__":
    np.random.seed(42)
    n_samples = 300
    weeks_training = np.random.uniform(1, 52, n_samples)
    sessions_per_week = np.random.randint(2, 7, n_samples)
    sleep_hours = np.random.uniform(4, 9, n_samples)
    protein_intake = np.random.uniform(50, 200, n_samples)
    X = np.column_stack([weeks_training, sessions_per_week, sleep_hours, protein_intake])
    actual_lift = (
        0.8 *weeks_training +
        6 *sessions_per_week +
        3 *sleep_hours +
        0.25* protein_intake +
        20
    )
    noise = np.random.normal(0, 5, n_samples)
    y = actual_lift + noise
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = default()
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transformed(X_test)

    print("training tym progress odel")
    model = LinearRegression(lr=0.1, iterations=1000)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    print_all_metrics(y_test, preds)

    print("Learned weights (weeks, sessions/week, sleep,protein):", model.weights.flatten().round(3))
    print("Learned bias:", round(model.bias, 3))
    new_lifter = np.array([[20, 5, 7, 150]])
    new_lifter_scaled = scaler.transformed(new_lifter)
    predicted_lift = model.predict(new_lifter_scaled)
    print("guessedmax lift for this lifter-", predicted_lift[0])