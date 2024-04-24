/*
Create users and databases for OneCX products

IMPORTANT!
  This script cannot be executed at once!
  Pick up the section you need in a SQL session and execute it manually.
  The DROP/CREATE database statements must be executed separately.
*/

DROP DATABASE IF EXISTS "onecx-chat";
  DROP ROLE IF EXISTS onecx_chat;
  CREATE USER onecx_chat WITH PASSWORD 'onecx_chat';
CREATE DATABASE "onecx-chat" WITH OWNER = onecx_chat;

