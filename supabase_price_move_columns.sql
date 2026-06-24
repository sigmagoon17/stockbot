alter table public.scan_history
    add column if not exists daily_move_pct numeric,
    add column if not exists five_day_move_pct numeric,
    add column if not exists move_vs_20d_vol numeric,
    add column if not exists unusual_move text;
