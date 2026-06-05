use std::env;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use anyhow::{Context, Result};
use zip::ZipArchive;

const BUFSZ: usize = 256 * 1024;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<()> {
    let urls: Vec<String> = env::args().skip(1).collect();
    if urls.is_empty() {
        anyhow::bail!("usage: downloader <url> [url] ...");
    }

    let total = urls.len();
    println!("starting {total} download(s)");

    let mut failures = 0u32;
    for (i, url) in urls.iter().enumerate() {
        println!("\n[{i}/{total}] {url}");

        let zip_path = match filename_from_url(url) {
            Some(n) => PathBuf::from(n),
            None => {
                eprintln!("  skip: could not derive filename");
                failures += 1;
                continue;
            }
        };

        let t0 = Instant::now();
        if let Err(e) = download(url, &zip_path) {
            eprintln!("  download failed: {e}");
            failures += 1;
            continue;
        }
        let size = fs::metadata(&zip_path).map(|m| m.len()).unwrap_or(0);
        println!(
            "  downloaded {} ({:.1} MB) in {:.1}s",
            zip_path.display(),
            size as f64 / 1_048_576.0,
            t0.elapsed().as_secs_f64()
        );

        let t0 = Instant::now();
        if let Err(e) = extract(&zip_path) {
            eprintln!("  extract failed: {e}");
            let _ = fs::remove_file(&zip_path);
            failures += 1;
            continue;
        }
        println!("  extracted in {:.1}s", t0.elapsed().as_secs_f64());

        if let Err(e) = fs::remove_file(&zip_path) {
            eprintln!("  warning: could not remove zip: {e}");
        }
    }

    println!("\n{total} file(s) processed, {failures} failed");
    if failures > 0 {
        anyhow::bail!("{failures}/{total} downloads failed");
    }
    Ok(())
}

fn download(url: &str, dest: &Path) -> Result<()> {
    eprintln!("  connecting...");
    let response = ureq::get(url).call().context("request failed")?;

    if let Some(len) = response.header("Content-Length").and_then(|v| v.parse::<u64>().ok()) {
        eprintln!("  content-length: {:.1} MB", len as f64 / 1_048_576.0);
    }

    let mut reader = BufReader::with_capacity(BUFSZ, response.into_reader());
    let mut file = BufWriter::with_capacity(
        BUFSZ,
        File::create(dest).with_context(|| format!("create {}", dest.display()))?,
    );
    io::copy(&mut reader, &mut file).context("write failed")?;
    file.flush()?;
    Ok(())
}

fn extract(zip_path: &Path) -> Result<()> {
    eprintln!("  opening zip...");
    let file = BufReader::with_capacity(
        BUFSZ,
        File::open(zip_path).with_context(|| format!("open {}", zip_path.display()))?,
    );
    let mut archive = ZipArchive::new(file).context("read zip")?;
    let n = archive.len();
    eprintln!("  {} entr{}", n, if n == 1 { "y" } else { "ies" });
    let base = zip_path.parent().unwrap_or(Path::new("."));

    for i in 0..n {
        let mut entry = archive.by_index(i)?;
        let Some(rel) = entry.enclosed_name() else {
            continue;
        };
        let outpath = base.join(rel);

        if entry.is_dir() {
            fs::create_dir_all(&outpath)
                .with_context(|| format!("mkdir {}", outpath.display()))?;
        } else {
            if let Some(parent) = outpath.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("mkdir {}", parent.display()))?;
            }
            let mut out = BufWriter::with_capacity(
                BUFSZ,
                File::create(&outpath)
                    .with_context(|| format!("create {}", outpath.display()))?,
            );
            io::copy(&mut entry, &mut out).context("write failed")?;
            out.flush()?;
        }
        println!("  -> {}", outpath.display());
    }
    Ok(())
}

fn filename_from_url(url: &str) -> Option<&str> {
    let (_, name) = url.rsplit_once('/')?;
    if name.is_empty() { None } else { Some(name) }
}
