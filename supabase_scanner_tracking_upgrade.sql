alter table public.scan_history
    add column if not exists scan_run_id text,
    add column if not exists setup_key text,
    add column if not exists scanner_version text,
    add column if not exists git_commit_sha text,
    add column if not exists raw_rank integer,
    add column if not exists diversified_rank integer,
    add column if not exists execution_rank integer,
    add column if not exists execution_selected boolean not null default false,
    add column if not exists selection_method text,
    add column if not exists first_seen_at timestamptz,
    add column if not exists last_seen_at timestamptz,
    add column if not exists times_recommended integer not null default 1,
    add column if not exists option_type text,
    add column if not exists raw_price_move_adjustment integer,
    add column if not exists effective_price_move_adjustment integer,
    add column if not exists base_score_without_price_move integer,
    add column if not exists entry_timestamp timestamptz,
    add column if not exists entry_price numeric,
    add column if not exists exit_timestamp timestamptz,
    add column if not exists exit_price numeric,
    add column if not exists exit_reason text,
    add column if not exists realized_pnl numeric,
    add column if not exists realized_return_on_risk numeric,
    add column if not exists closing_underlying_price numeric,
    add column if not exists days_held integer,
    add column if not exists maximum_favorable_excursion numeric,
    add column if not exists maximum_adverse_excursion numeric,
    add column if not exists last_update_error text,
    add column if not exists update_retryable boolean not null default false;

create index if not exists scan_history_scan_run_id_idx
    on public.scan_history(scan_run_id);

create index if not exists scan_history_setup_key_idx
    on public.scan_history(setup_key);

create index if not exists scan_history_execution_selected_idx
    on public.scan_history(execution_selected, scan_time desc);

alter table public.alpaca_paper_orders
    add column if not exists scan_run_id text,
    add column if not exists setup_key text,
    add column if not exists execution_rank integer,
    add column if not exists selection_method text,
    add column if not exists ticker_score integer,
    add column if not exists quant_score integer,
    add column if not exists max_profit numeric,
    add column if not exists max_risk numeric,
    add column if not exists entry_timestamp timestamptz,
    add column if not exists entry_price numeric,
    add column if not exists opening_order_status text,
    add column if not exists opening_filled_at timestamptz,
    add column if not exists opening_filled_avg_price numeric,
    add column if not exists spread_width_per_share numeric,
    add column if not exists filled_max_profit_per_share numeric,
    add column if not exists filled_max_risk_per_share numeric,
    add column if not exists fill_validation_error text,
    add column if not exists exit_policy text not null default 'none',
    add column if not exists position_status text not null default 'pending',
    add column if not exists exit_signal_time timestamptz,
    add column if not exists exit_reason text,
    add column if not exists close_order_id text,
    add column if not exists close_client_order_id text,
    add column if not exists close_order_status text,
    add column if not exists close_order_submitted_at timestamptz,
    add column if not exists close_attempt_count integer not null default 0,
    add column if not exists last_close_attempt_at timestamptz,
    add column if not exists close_attempt_run_id text,
    add column if not exists exit_retryable boolean not null default true,
    add column if not exists close_attempt_history jsonb not null default '[]'::jsonb,
    add column if not exists exit_fill_time timestamptz,
    add column if not exists exit_fill_price numeric,
    add column if not exists realized_pnl numeric,
    add column if not exists realized_return_on_risk numeric,
    add column if not exists maximum_favorable_excursion numeric,
    add column if not exists maximum_adverse_excursion numeric,
    add column if not exists last_exit_error text;

alter table public.alpaca_paper_orders
    alter column position_status set default 'pending';

drop index if exists public.alpaca_paper_orders_leg_key_idx;

create unique index alpaca_paper_orders_leg_key_idx
    on public.alpaca_paper_orders(leg_key)
    where leg_key is not null
      and position_status in ('pending', 'open');

create unique index if not exists alpaca_paper_orders_close_client_order_id_idx
    on public.alpaca_paper_orders(close_client_order_id)
    where close_client_order_id is not null;

create index if not exists alpaca_paper_orders_scan_run_id_idx
    on public.alpaca_paper_orders(scan_run_id);

create index if not exists alpaca_paper_orders_setup_key_idx
    on public.alpaca_paper_orders(setup_key);

drop function if exists public.claim_alpaca_paper_exit(
    bigint, text, timestamptz, text
);

create or replace function public.claim_alpaca_paper_exit(
    p_order_id bigint,
    p_exit_reason text,
    p_signal_time timestamptz,
    p_close_client_order_id text,
    p_attempt_number integer,
    p_attempt_run_id text
)
returns setof public.alpaca_paper_orders
language plpgsql
security definer
set search_path = public
as $$
begin
    return query
    update public.alpaca_paper_orders
    set
        exit_signal_time = p_signal_time,
        exit_reason = p_exit_reason,
        close_client_order_id = p_close_client_order_id,
        close_order_status = 'submitting',
        close_order_id = null,
        close_order_submitted_at = null,
        exit_fill_time = null,
        exit_fill_price = null,
        close_attempt_history = case
            when coalesce(close_attempt_count, 0) > 0 then
                coalesce(close_attempt_history, '[]'::jsonb)
                || jsonb_build_array(
                    jsonb_build_object(
                        'attempt', close_attempt_count,
                        'order_id', close_order_id,
                        'client_order_id', close_client_order_id,
                        'status', close_order_status,
                        'submitted_at', close_order_submitted_at,
                        'error', last_exit_error
                    )
                )
            else coalesce(close_attempt_history, '[]'::jsonb)
        end,
        close_attempt_count = p_attempt_number,
        last_close_attempt_at = p_signal_time,
        close_attempt_run_id = p_attempt_run_id,
        exit_retryable = false,
        last_exit_error = null
    where id = p_order_id
      and position_status = 'open'
      and opening_order_status = 'filled'
      and opening_filled_avg_price is not null
      and coalesce(close_attempt_count, 0) + 1 = p_attempt_number
      and close_attempt_run_id is distinct from p_attempt_run_id
      and (
          close_order_status is null
          or (
              lower(close_order_status) in ('rejected', 'canceled', 'expired')
              and coalesce(exit_retryable, false)
          )
      )
    returning *;
end;
$$;

revoke all on function public.claim_alpaca_paper_exit(
    bigint, text, timestamptz, text, integer, text
)
    from public, anon, authenticated;
grant execute on function public.claim_alpaca_paper_exit(
    bigint, text, timestamptz, text, integer, text
)
    to service_role;

alter table public.alpaca_paper_position_snapshots
    add column if not exists exit_policy text,
    add column if not exists target_value_per_share numeric,
    add column if not exists current_value_per_share numeric,
    add column if not exists exit_signal text;

create or replace view public.scan_history_setup_summary as
select
    setup_key,
    min(scan_time) as first_seen_at,
    max(scan_time) as last_seen_at,
    count(*) as recommendation_occurrences,
    count(distinct scan_run_id) as scan_runs,
    max(setup_score) as best_setup_score,
    bool_or(coalesce(execution_selected, false)) as ever_execution_selected
from public.scan_history
where setup_key is not null
group by setup_key;
