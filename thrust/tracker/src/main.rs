use clap::Parser;
use tracker::cli;

fn main() -> std::process::ExitCode {
    let cli = cli::Cli::parse();
    let json = cli.machine();
    match cli::run(cli) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            if json {
                // Machine callers get the failure on stdout, in the shape they parse.
                println!("{}", serde_json::json!({ "error": { "code": e.code(), "message": e.to_string() } }));
            } else {
                eprintln!("tracker: {e}");
            }
            std::process::ExitCode::FAILURE
        }
    }
}
