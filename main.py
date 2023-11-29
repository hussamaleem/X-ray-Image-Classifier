import optuna_class
import config
from datetime import datetime


time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

runner = optuna_class.OptunaOptim(time_stamp,
                                  config.device)

runner.run_objective()

