/*
 Navicat Premium Data Transfer

 Source Server: 127.0.0.1
 Source Server Type: MySQL
 Source Server Version: 80032
 Source Host: localhost:3306
 Source Schema: flaskdemo_user

 Target Server Type: MySQL
 Target Server Version: 80032
 File Encoding         : 65001

 Date: 25/10/2024 20:55:06
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for users
-- ----------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users`  (
  `id` int NOT NULL AUTO_INCREMENT,
  `username` varchar(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  `password` varchar(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL,
  PRIMARY KEY (`id`) USING BTREE,
  UNIQUE INDEX `username`(`username`) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of users
-- ----------------------------
INSERT INTO `users` VALUES (1, 'admin', 'admin');
INSERT INTO `users` VALUES (2, 'amiya', 'scrypt:32768:8:1$dKCN48RcWV4JR8j4$15d0b6ec14048bd7b4989b443137eb8ffdc250fe79e11ffca1502ec99e60ec6eeab859e099f0a74a4ba1e7c78ee3a7898ee893cd79d1c1ffbe1cf999a220fb58');

SET FOREIGN_KEY_CHECKS = 1;
