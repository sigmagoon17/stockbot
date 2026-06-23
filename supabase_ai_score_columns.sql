alter table public.scan_history
    add column if not exists quant_score integer,
    add column if not exists event_adjustment integer,
    add column if not exists event_label text,
    add column if not exists event_confidence text,
    add column if not exists event_summary text;
