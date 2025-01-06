// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2024 Oxford Quantum Circuits Ltd

#![warn(clippy::all, clippy::pedantic)]
#![allow(
  non_upper_case_globals,
  non_snake_case,
  unused_macros,
  dead_code,
  unused_variables,
  unused_must_use
)]
#![allow(clippy::needless_pass_by_value)]
#![feature(get_mut_unchecked)]
#![feature(unwrap_infallible)]
#![feature(strict_provenance)]
#![feature(is_sorted)]
extern crate core;

use log::{log, log_enabled, Level, LevelFilter};
use std::env;
use std::env::current_exe;
use std::fs;
use std::fs::File;
use std::path::Path;

mod analysis;
mod builders;
mod config;
mod evaluator;
mod exceptions;
mod execution;
mod features;
mod graphs;
mod hardware;
mod instructions;
mod python;
mod runtime;
mod smart_pointers;

const DEFAULT_LOG_FILE: &str = "rasqal_logs.txt";

const DEFAULT_LOG_FOLDER: &str = ".logs";

/// Native initialization of the loggers. Defaults to executable position if deployed, if it
/// detects it's in development mode it'll move log file back up the folder tree.
#[ctor::ctor]
fn native_logger_initialize() {
  // If we're in CI just print everything to the commandline.
  if let Ok(_) = env::var("CI") {
    initialize_commandline_logging(Vec::new());
    return;
  }

  let path = if let Ok(val) = current_exe() {
    // If we're embedded we need to be given a different file path to log too.
    if val.ends_with("python.exe") {
      return;
    }

    let current_folder = val.parent().unwrap();

    // Walk back to root rasqal folder if we're in a build, otherwise at that folder level.
    if current_folder.ends_with("deps") {
      Some(
        current_folder
          .parent()
          .unwrap()
          .parent()
          .unwrap()
          .parent()
          .unwrap()
          .parent()
          .unwrap()
          .join(DEFAULT_LOG_FOLDER)
          .join(DEFAULT_LOG_FILE)
          .to_str()
          .unwrap()
          .to_string()
      )
    } else {
      Some(
        current_folder
          .join(DEFAULT_LOG_FOLDER)
          .join(DEFAULT_LOG_FILE)
          .to_str()
          .unwrap()
          .to_string()
      )
    }
  } else {
    None
  };

  initialize_loggers(path);
  log!(Level::Info, "Initialized on library startup.");
}

fn initialize_commandline_logging(appended_messages: Vec<String>) {
  // If we're fallen through previous forms of logger init have failed.
  env_logger::builder()
    .filter_level(LevelFilter::Debug)
    .init();

  log!(Level::Info, "Commandline logging initialized.");
  for val in appended_messages {
    log!(Level::Info, "{}", val);
  }
}

fn initialize_loggers(log_path: Option<String>) {
  // If we've already been enabled, just do nothing.
  if log_enabled!(Level::Error) {
    return;
  }

  let mut appended_messages = Vec::new();
  if let Some(logging_path) = log_path {
    let log_file = Path::new(logging_path.as_str());
    if let Some(dir) = log_file.parent() {
      fs::create_dir_all(dir);
    }

    let file = File::create(log_file);
    if let Ok(file) = file {
      // TODO: Just print to both commandline and file.
      let target = Box::new(file);
      env_logger::builder()
        .target(env_logger::Target::Pipe(target))
        .filter_level(LevelFilter::Debug)
        .format_suffix("\n")
        .init();

      log!(Level::Info, "File logging initialized.");
      return;
    }

    appended_messages.push(format!(
      "Attempted to open file at {} to log, failed with: {}",
      logging_path,
      file.err().unwrap()
    ));
  }

  // If we're fallen through previous forms of logger init have failed.
  initialize_commandline_logging(appended_messages);
}
