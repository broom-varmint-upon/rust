use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use flate2::write::GzEncoder;
use flate2::Compression;
use jetstreamer::firehose::{BlockData, TransactionData};
use jetstreamer::plugin::{Plugin, PluginFuture};
use serde::Serialize;

#[derive(Parser)]
#[command(name = "solana-fetcher")]
#[command(about = "Fetch Solana blockchain data from Old Faithful via Jetstreamer")]
struct Cli {
    start_slot: u64,
    end_slot: u64,
    output_dir: PathBuf,
    #[arg(short, long, default_value = "2")]
    threads: usize,
}

#[derive(Serialize)]
struct BlockRecord {
    r#type: String,
    slot: u64,
    parent_slot: u64,
    blockhash: String,
    parent_blockhash: String,
    block_time: Option<i64>,
    block_height: Option<u64>,
    executed_transaction_count: u64,
    entry_count: u64,
}

#[derive(Serialize)]
struct TransactionRecord {
    r#type: String,
    slot: u64,
    index: usize,
    signature: String,
    is_vote: bool,
    meta: TransactionMetaRecord,
    message: TransactionMessageRecord,
}

#[derive(Serialize)]
struct TransactionMetaRecord {
    err: Option<String>,
    fee: u64,
    pre_balances: Vec<u64>,
    post_balances: Vec<u64>,
    log_messages: Vec<String>,
    inner_instructions: Vec<InnerInstructionRecord>,
}

#[derive(Serialize)]
struct InnerInstructionRecord {
    index: u32,
    instructions: Vec<InstructionRecord>,
}

#[derive(Serialize)]
struct InstructionRecord {
    program_id_index: u8,
    accounts: Vec<u8>,
    data: String,
}

#[derive(Serialize)]
struct TransactionMessageRecord {
    account_keys: Vec<String>,
    recent_blockhash: String,
    instructions: Vec<InstructionRecord>,
}

#[derive(Serialize)]
struct SkippedSlotRecord {
    r#type: String,
    slot: u64,
}

struct JsonExportPlugin {
    writer: Mutex<Option<GzEncoder<BufWriter<File>>>>,
    block_count: AtomicU64,
    tx_count: AtomicU64,
    start_time: Instant,
    output_path: PathBuf,
}

impl JsonExportPlugin {
    fn new(output_dir: &std::path::Path, start_slot: u64, end_slot: u64) -> Result<Self> {
        fs::create_dir_all(output_dir)?;
        let filename = format!("blocks_{}_{}.ndjson.gz", start_slot, end_slot);
        let output_path = output_dir.join(&filename);
        let file = File::create(&output_path)
            .with_context(|| format!("create {}", output_path.display()))?;
        let buf = BufWriter::with_capacity(256 * 1024, file);
        let encoder = GzEncoder::new(buf, Compression::fast());

        Ok(Self {
            writer: Mutex::new(Some(encoder)),
            block_count: AtomicU64::new(0),
            tx_count: AtomicU64::new(0),
            start_time: Instant::now(),
            output_path,
        })
    }

    fn write_line(&self, record: &impl Serialize) -> Result<()> {
        let mut guard = self.writer.lock().unwrap();
        let encoder = guard.as_mut().context("writer closed")?;
        let mut line = serde_json::to_vec(record)?;
        line.push(b'\n');
        encoder.write_all(&line)?;
        Ok(())
    }

    fn flush_and_close(&self) -> Result<()> {
        let mut guard = self.writer.lock().unwrap();
        if let Some(encoder) = guard.take() {
            let mut buf = encoder.finish()?;
            buf.flush()?;
        }
        Ok(())
    }

    fn print_stats(&self) {
        let blocks = self.block_count.load(Ordering::Relaxed);
        let txs = self.tx_count.load(Ordering::Relaxed);
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let tps = if elapsed > 0.0 { txs as f64 / elapsed } else { 0.0 };
        let bps = if elapsed > 0.0 { blocks as f64 / elapsed } else { 0.0 };
        println!(
            "=== Stats: {} blocks, {} txs, {:.1}s elapsed, {:.0} blocks/s, {:.0} tx/s ===",
            blocks, txs, elapsed, bps, tps
        );
    }
}

impl Plugin for JsonExportPlugin {
    fn name(&self) -> &'static str {
        "json-export"
    }

    fn on_block<'a>(
        &'a self,
        _thread_id: usize,
        _db: Option<Arc<clickhouse::Client>>,
        block: &'a BlockData,
    ) -> PluginFuture<'a> {
        Box::pin(async move {
            match block {
                BlockData::Block {
                    parent_slot,
                    parent_blockhash,
                    slot,
                    blockhash,
                    block_time,
                    block_height,
                    executed_transaction_count,
                    entry_count,
                    ..
                } => {
                    let record = BlockRecord {
                        r#type: "block".to_string(),
                        slot: *slot,
                        parent_slot: *parent_slot,
                        blockhash: blockhash.to_string(),
                        parent_blockhash: parent_blockhash.to_string(),
                        block_time: *block_time,
                        block_height: *block_height,
                        executed_transaction_count: *executed_transaction_count,
                        entry_count: *entry_count,
                    };
                    if let Err(e) = self.write_line(&record) {
                        log::error!("write block failed: {e}");
                    }
                    self.block_count.fetch_add(1, Ordering::Relaxed);
                }
                BlockData::PossibleLeaderSkipped { slot } => {
                    let record = SkippedSlotRecord {
                        r#type: "skipped".to_string(),
                        slot: *slot,
                    };
                    if let Err(e) = self.write_line(&record) {
                        log::error!("write skipped slot failed: {e}");
                    }
                }
            }
            Ok(())
        })
    }

    fn on_transaction<'a>(
        &'a self,
        _thread_id: usize,
        _db: Option<Arc<clickhouse::Client>>,
        tx: &'a TransactionData,
    ) -> PluginFuture<'a> {
        Box::pin(async move {
            let meta = &tx.transaction_status_meta;

            let err_str = meta.status.err().map(|e| format!("{e:?}"));
            let log_messages = meta.log_messages.clone();

            let inner_instructions: Vec<InnerInstructionRecord> = meta
                .inner_instructions
                .iter()
                .map(|ii| InnerInstructionRecord {
                    index: ii.index,
                    instructions: ii
                        .instructions
                        .iter()
                        .map(|i| InstructionRecord {
                            program_id_index: i.program_id_index,
                            accounts: i.accounts.clone(),
                            data: bs58::encode(&i.data).into_string(),
                        })
                        .collect(),
                })
                .collect();

            let message = &tx.transaction.message;
            let account_keys: Vec<String> = message
                .static_account_keys()
                .iter()
                .map(|k| k.to_string())
                .collect();

            let instructions: Vec<InstructionRecord> = message
                .instructions()
                .iter()
                .map(|i| InstructionRecord {
                    program_id_index: i.program_id_index,
                    accounts: i.accounts.clone(),
                    data: bs58::encode(&i.data).into_string(),
                })
                .collect();

            let recent_blockhash = message.recent_blockhash().to_string();

            let record = TransactionRecord {
                r#type: "transaction".to_string(),
                slot: tx.slot,
                index: tx.transaction_slot_index,
                signature: tx.signature.to_string(),
                is_vote: tx.is_vote,
                meta: TransactionMetaRecord {
                    err: err_str,
                    fee: meta.fee,
                    pre_balances: meta.pre_balances.clone(),
                    post_balances: meta.post_balances.clone(),
                    log_messages,
                    inner_instructions,
                },
                message: TransactionMessageRecord {
                    account_keys,
                    recent_blockhash,
                    instructions,
                },
            };

            if let Err(e) = self.write_line(&record) {
                log::error!("write tx failed: {e}");
            }
            self.tx_count.fetch_add(1, Ordering::Relaxed);

            let tx_count = self.tx_count.load(Ordering::Relaxed);
            if tx_count % 10000 == 0 {
                self.print_stats();
            }

            Ok(())
        })
    }

    fn on_exit(&self, _db: Option<Arc<clickhouse::Client>>) -> PluginFuture<'_> {
        Box::pin(async move {
            if let Err(e) = self.flush_and_close() {
                log::error!("flush failed: {e}");
            }
            self.print_stats();
            println!("Output: {}", self.output_path.display());
            Ok(())
        })
    }
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    println!(
        "Fetching slots {}..{} with {} threads",
        cli.start_slot, cli.end_slot, cli.threads
    );

    let plugin = JsonExportPlugin::new(&cli.output_dir, cli.start_slot, cli.end_slot)?;

    let plugin = Box::new(plugin);

    jetstreamer::JetstreamerRunner::new()
        .with_plugin(plugin)
        .with_threads(cli.threads)
        .with_slot_range_bounds(cli.start_slot, cli.end_slot)
        .with_clickhouse_dsn("off")
        .run()
        .context("jetstreamer runner failed")?;

    Ok(())
}
