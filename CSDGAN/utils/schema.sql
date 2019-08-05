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
  end_time TIMESTAMP,
  format VARCHAR(20) NOT NULL,
  filesize INTEGER NOT NULL,
  depvar VARCHAR(50),
  num_augs INTEGER NOT NULL DEFAULT 0,
  live TINYINT(1) NOT NULL DEFAULT 1,
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

insert into status_info
  values
  (1, 'Not started'),
  (2, 'Preprocessing data'),
  (3, 'Training in progress...0/4'),
  (4, 'Training in progress...1/4'),
  (5, 'Training in progress...1/2'),
  (6, 'Training in progress...3/4'),
  (7, 'Training complete - Generating data'),
  (8, 'Complete - Data available'),
  (99, 'Error - Run failed'),
  (100, 'No Longer Available');
