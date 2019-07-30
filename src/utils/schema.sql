drop table IF EXISTS user;
drop table IF EXISTS run;
drop table IF EXISTS status;
drop table IF EXISTS status_info;
drop table IF EXISTS post;

create TABLE user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT UNIQUE NOT NULL,
  password TEXT NOT NULL,
  last_login TIMESTAMP NOT NULL,
  num_logins INTEGER NOT NULL
);

create TABLE run (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  user_id INTEGER NOT NULL,
  start_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  end_time TIMESTAMP,
  format TEXT NOT NULL,
  filesize INTEGER NOT NULL,
  FOREIGN KEY (user_id) REFERENCES user (id)
);

create TABLE status (
  run_id INTEGER NOT NULL,
  status_id INTEGER NOT NULL,
  update_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, status_id),
  FOREIGN KEY (run_id) REFERENCES run (id)
  FOREIGN KEY (status_id) REFERENCES status_info (id)
);

create TABLE status_info (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  descr TEXT NOT NULL
);

insert into status_info (descr)
  values
  ('Not started'),  -- 1
  ('Preprocessing data'),  -- 2
  ('Training in progress...0/4'),  -- 3
  ('Training in progress...1/4'),  -- 4
  ('Training in progress...1/2'),  -- 5
  ('Training in progress...3/4'),  -- 6
  ('Training complete - Generating data'),  -- 7
  ('Complete - Data available'),  -- 8
  ('Error - Run failed');  -- 9

create TABLE post (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  author_id INTEGER NOT NULL,
  created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  FOREIGN KEY (author_id) REFERENCES user (id)
);