SET FOREIGN_KEY_CHECKS=0;
drop table IF EXISTS user, run, status, status_info;
SET FOREIGN_KEY_CHECKS=1;

create TABLE user (
  id INTEGER PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(20) UNIQUE NOT NULL,
  password VARCHAR(100) NOT NULL,
  last_login TIMESTAMP NOT NULL,
  num_logins INTEGER NOT NULL
);

create TABLE run (
  id INTEGER PRIMARY KEY AUTO_INCREMENT,
  title VARCHAR(20) NOT NULL,
  user_id INTEGER NOT NULL,
  start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  end_time TIMESTAMP NULL DEFAULT NULL,
  format VARCHAR(20) NOT NULL,
  filesize INTEGER,
  depvar VARCHAR(50),
  cont_inputs VARCHAR(200),
  benchmark FLOAT DEFAULT NULL,
  num_augs INTEGER NOT NULL DEFAULT 0,
  num_retrains INTEGER NOT NULL DEFAULT 0,
  live TINYINT(1) NOT NULL DEFAULT 1,
  data_job_id VARCHAR(36) DEFAULT NULL,
  train_job_id VARCHAR(36) DEFAULT NULL,
  generate_job_id VARCHAR(36) DEFAULT NULL,
  FOREIGN KEY (user_id) REFERENCES user (id)
);

create TABLE status_info (
  id INTEGER PRIMARY KEY,
  descr VARCHAR(50) NOT NULL
);

create TABLE status (
  run_id INTEGER NOT NULL,
  status_id INTEGER NOT NULL,
  update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, status_id),
  FOREIGN KEY (run_id) REFERENCES run (id),
  FOREIGN KEY (status_id) REFERENCES status_info (id)
);

-- Make sure to check constants.py as well if changes are made
insert into status_info
  values
  (1, 'Not started'),
  (2, 'Kicked off'),
  (3, 'Preprocessing data'),
  (4, 'Benchmarking'),
  (5, 'Training in progress...0/4'),
  (6, 'Training in progress...1/4'),
  (7, 'Training in progress...1/2'),
  (8, 'Training in progress...3/4'),
  (9, 'Training complete - Generating data'),
  (10, 'Complete - Data available'),
  (11, 'Retrain 0/4'),
  (12, 'Retrain 1/4'),
  (13, 'Retrain 2/4'),
  (14, 'Retrain 3/4'),
  (15, 'Retraining Complete'),
  (98, 'Forcefully exited early by user'),
  (99, 'Error - Run failed'),
  (100, 'No Longer Available');
