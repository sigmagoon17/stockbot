alter table public.alpaca_paper_orders
    add column if not exists manual_override boolean not null default false,
    add column if not exists overridden_filters text[],
    add column if not exists original_rejection_reasons text[],
    add column if not exists override_timestamp timestamptz,
    add column if not exists original_quantitative_score integer;

create index if not exists alpaca_paper_orders_manual_override_idx
    on public.alpaca_paper_orders(manual_override, scan_time desc);
