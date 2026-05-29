import Database from "better-sqlite3";
import { mkdirSync } from "fs";
import { dirname } from "path";

let dbInstance: Database.Database | null = null;

export function getCorrectionsDbPath(): string {
  return process.env.CORRECTIONS_DB_PATH ?? "data/corrections.db";
}

export function getDb(): Database.Database {
  if (dbInstance) return dbInstance;
  const path = getCorrectionsDbPath();
  mkdirSync(dirname(path), { recursive: true });
  const db = new Database(path);
  db.exec(`
    CREATE TABLE IF NOT EXISTS corrections (
      id TEXT PRIMARY KEY,
      created_at INTEGER NOT NULL,
      case_canonical TEXT NOT NULL,
      agent_pulpal TEXT NOT NULL,
      agent_apical TEXT NOT NULL,
      corrected_pulpal TEXT NOT NULL,
      corrected_apical TEXT NOT NULL,
      reasoning TEXT NOT NULL,
      misunderstood TEXT,
      embed_document TEXT NOT NULL,
      embedding_dim INTEGER NOT NULL,
      embedding BLOB NOT NULL
    );
    CREATE INDEX IF NOT EXISTS corrections_created_at ON corrections(created_at DESC);
  `);
  dbInstance = db;
  return db;
}
