alter table public.scan_history
    add column if not exists price_move_adjustment integer,
    add column if not exists move_setup text;
