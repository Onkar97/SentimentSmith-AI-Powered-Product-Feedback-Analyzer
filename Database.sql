CREATE DATABASE IF NOT EXISTS pythonlogin DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;
USE pythonlogin;

CREATE TABLE IF NOT EXISTS accounts (
	id int(11) NOT NULL AUTO_INCREMENT,
  	username varchar(50) NOT NULL,
  	password varchar(255) NOT NULL,
  	email varchar(100) NOT NULL,
    PRIMARY KEY (id)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8;

INSERT INTO accounts (id, username, password, email) VALUES (1, 'test', 'test', 'test@test.com');


CREATE USER 'onkar'@'localhost' IDENTIFIED BY '';
GRANT ALL PRIVILEGES ON *.* TO 'onkar'@'localhost';
UPDATE user SET plugin='auth_socket' WHERE User='onkar';
FLUSH PRIVILEGES;
exit;

sudo service mysql restart
update user set plugin="mysqlnativepassword" where user='onkar';


CREATE TABLE product (
	id int(11) NOT NULL AUTO_INCREMENT,
	user_name VARCHAR(255) NOT NULL,
	product_name varchar(50) NOT NULL,
  	product_size varchar(255) NOT NULL,
  	product_colour varchar(255) NOT NULL,
  	product_shape varchar(255) NOT NULL,
  	add_info varchar(255),
  	product_path VARCHAR(255) NOT NULL,
  PRIMARY KEY (id)
);

CREATE TABLE product_reviews (
	id int(11) NOT NULL AUTO_INCREMENT,
	product_name VARCHAR(255) NOT NULL,
  	user_email varchar(255) NOT NULL,
  	product_rating varchar(255) NOT NULL,
  	product_text varchar(255) NOT NULL,
  	vote varchar(255) NOT NULL,
  PRIMARY KEY (id)
);
