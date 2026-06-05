use std::env;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use anyhow::{Context, Result};
use zip::ZipArchive;

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
    let url = env::args()
        .nth(1)
        .context("usage: downloader <url>")?;

    let zip_path = PathBuf::from(
        filename_from_url(&url).context("could not derive filename from url")?,
    );

    download(&url, &zip_path).context("download failed")?;
    println!("downloaded {}", zip_path.display());

    if let Err(e) = extract(&zip_path).context("extraction failed") {
        let _ = fs::remove_file(&zip_path);
        return Err(e);
    }

    fs::remove_file(&zip_path).context("could not remove zip")?;
    println!("removed {}", zip_path.display());

    Ok(())
}

fn download(url: &str, dest: &Path) -> Result<()> {
    let response = ureq::get(url).call().context("request failed")?;
    let mut reader = response.into_reader();
    let mut file = File::create(dest).with_context(|| format!("create {}", dest.display()))?;
    io::copy(&mut reader, &mut file).context("write failed")?;
    Ok(())
}

fn extract(zip_path: &Path) -> Result<()> {
    let file = File::open(zip_path).with_context(|| format!("open {}", zip_path.display()))?;
    let mut archive = ZipArchive::new(file).context("read zip")?;
    let base = zip_path.parent().unwrap_or(Path::new("."));

    for i in 0..archive.len() {
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
            let mut out =
                File::create(&outpath).with_context(|| format!("create {}", outpath.display()))?;
            io::copy(&mut entry, &mut out).context("write failed")?;
        }
        println!("extracted {}", outpath.display());
    }
    Ok(())
}

fn filename_from_url(url: &str) -> Option<&str> {
    let (_, name) = url.rsplit_once('/')?;
    if name.is_empty() { None } else { Some(name) }
}
