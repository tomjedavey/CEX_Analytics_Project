WITH wallet_activity AS (
    SELECT 
        from_address AS wallet,
        COUNT(*) AS tx_count
    FROM polygon_onchain_core_data.core.fact_transactions
    WHERE block_timestamp >= DATEADD(MONTH, -6, CURRENT_DATE())
    GROUP BY from_address
),

sampled_wallets AS (
    (
        SELECT wallet FROM wallet_activity WHERE tx_count >= 100 LIMIT 10000
    )
    UNION ALL
    (
        SELECT wallet FROM wallet_activity WHERE tx_count BETWEEN 10 AND 99 LIMIT 20000
    )
    UNION ALL
    (
        SELECT wallet FROM wallet_activity WHERE tx_count < 10 LIMIT 20000
    )
),

date_filtered_txns AS (
    SELECT 
        t.from_address AS wallet,
        t.block_timestamp::DATE AS tx_date,
        t.tx_hash,
        t.origin_function_signature,
        t.to_address
    FROM polygon_onchain_core_data.core.fact_transactions t
    JOIN sampled_wallets w ON t.from_address = w.wallet
    WHERE t.block_timestamp >= DATEADD(MONTH, -6, CURRENT_DATE())
),

monthly_tx_freq AS (
    SELECT 
        wallet,
        COUNT(DISTINCT tx_hash) / 6.0 AS tx_per_month
    FROM date_filtered_txns
    GROUP BY wallet
),

token_transfers AS (
    SELECT 
        t.origin_from_address AS wallet,
        t.contract_address,
        t.block_timestamp::DATE AS tx_date,
        t.amount_usd,
        t.origin_function_signature,
        t.origin_to_address
    FROM polygon_onchain_core_data.core.ez_token_transfers t
    JOIN sampled_wallets w ON t.origin_from_address = w.wallet
    WHERE t.block_timestamp >= DATEADD(MONTH, -6, CURRENT_DATE())
),

token_diversity AS (
    SELECT 
        wallet,
        COUNT(DISTINCT contract_address) AS unique_tokens
    FROM token_transfers
    GROUP BY wallet
),

protocol_diversity AS (
    SELECT 
        wallet,
        COUNT(DISTINCT to_address) AS unique_contracts
    FROM date_filtered_txns
    WHERE to_address IS NOT NULL
    GROUP BY wallet
),

transfer_usd AS (
    SELECT 
        wallet,
        COALESCE(SUM(amount_usd), 0) AS total_transfer_usd
    FROM token_transfers
    GROUP BY wallet
),

interaction_diversity AS (
    SELECT 
        wallet,
        COUNT(DISTINCT origin_function_signature) AS unique_methods
    FROM date_filtered_txns
    WHERE origin_function_signature IS NOT NULL
    GROUP BY wallet
),

activity_range AS (
    SELECT 
        wallet,
        DATEDIFF('day', MIN(tx_date), MAX(tx_date)) AS active_days
    FROM date_filtered_txns
    GROUP BY wallet
),

transfer_value_stats AS (
    SELECT 
        wallet,
        AVG(amount_usd) AS avg_transfer_usd,
        STDDEV_POP(amount_usd) AS usd_transfer_stddev
    FROM token_transfers
    WHERE amount_usd IS NOT NULL
    GROUP BY wallet
),

wallet_event_summary AS (
    SELECT
        tx.from_address AS wallet,
        labels.label_type
    FROM polygon_onchain_core_data.core.fact_transactions tx
    INNER JOIN polygon_onchain_core_data.core.dim_labels labels 
        ON tx.to_address = labels.address
    WHERE tx.block_timestamp >= DATEADD(MONTH, -6, CURRENT_DATE())
      AND tx.from_address IN (SELECT wallet FROM sampled_wallets)
),

wallet_intent_counts AS (
    SELECT 
        wallet,
        SUM(CASE WHEN label_type = 'dex' THEN 1 ELSE 0 END) AS dex_events,
        SUM(CASE WHEN label_type = 'games' THEN 1 ELSE 0 END) AS games_events,
        SUM(CASE WHEN label_type = 'cex' THEN 1 ELSE 0 END) AS cex_events,
        SUM(CASE WHEN label_type = 'dapp' THEN 1 ELSE 0 END) AS dapp_events,
        SUM(CASE WHEN label_type = 'chadmin' THEN 1 ELSE 0 END) AS chadmin_events,
        SUM(CASE WHEN label_type = 'defi' THEN 1 ELSE 0 END) AS defi_events,
        SUM(CASE WHEN label_type = 'bridge' THEN 1 ELSE 0 END) AS bridge_events,
        SUM(CASE WHEN label_type = 'nft' THEN 1 ELSE 0 END) AS nft_events, 
        SUM(CASE WHEN label_type = 'token' THEN 1 ELSE 0 END) AS token_events,
        SUM(CASE WHEN label_type = 'flotsam' THEN 1 ELSE 0 END) AS flotsam_events
    FROM wallet_event_summary
    GROUP BY wallet
),

bridge_behavior AS (
    SELECT 
        t.origin_from_address AS wallet,
        COUNT(CASE WHEN t.origin_to_address = l.address THEN 1 END) AS bridge_outflow_count,
        COUNT(CASE WHEN t.origin_from_address = l.address THEN 1 END) AS bridge_inflow_count,
        SUM(CASE WHEN t.origin_to_address = l.address OR t.origin_from_address = l.address THEN t.amount_usd ELSE 0 END) AS bridge_total_volume_usd
    FROM polygon_onchain_core_data.core.ez_token_transfers t
    INNER JOIN polygon_onchain_core_data.core.dim_labels l 
        ON (t.origin_to_address = l.address OR t.origin_from_address = l.address)
    WHERE l.label_type = 'bridge'
      AND t.block_timestamp >= DATEADD(MONTH, -6, CURRENT_DATE())
      AND (t.origin_from_address IN (SELECT wallet FROM sampled_wallets)
           OR t.origin_to_address IN (SELECT wallet FROM sampled_wallets))
    GROUP BY wallet
),

wallet_features AS (
    SELECT 
        tx.wallet,
        tx.tx_per_month,
        COALESCE(tok.unique_tokens, 0) AS token_diversity,
        COALESCE(proto.unique_contracts, 0) AS protocol_diversity,
        COALESCE(transfer.total_transfer_usd, 0) AS total_transfer_usd,
        COALESCE(interact.unique_methods, 0) AS interaction_diversity,
        COALESCE(act_range.active_days, 0) AS active_duration_days,
        COALESCE(value_stats.avg_transfer_usd, 0) AS avg_transfer_usd,
        COALESCE(value_stats.usd_transfer_stddev, 0) AS usd_transfer_stddev,

        COALESCE(intent.dex_events, 0) AS dex_events,
        COALESCE(intent.games_events, 0) AS games_events,
        COALESCE(intent.cex_events, 0) AS cex_events,
        COALESCE(intent.dapp_events, 0) AS dapp_events,
        COALESCE(intent.chadmin_events, 0) AS chadmin_events,
        COALESCE(intent.defi_events, 0) AS defi_events,
        COALESCE(intent.bridge_events, 0) AS bridge_events,
        COALESCE(intent.nft_events, 0) AS nft_events,
        COALESCE(intent.token_events, 0) AS token_events,
        COALESCE(intent.flotsam_events, 0) AS flotsam_events,

        COALESCE(bridge.bridge_outflow_count, 0) AS bridge_outflow_count,
        COALESCE(bridge.bridge_inflow_count, 0) AS bridge_inflow_count,
        COALESCE(bridge.bridge_total_volume_usd, 0) AS bridge_total_volume_usd
    FROM monthly_tx_freq tx
    LEFT JOIN token_diversity tok ON tx.wallet = tok.wallet
    LEFT JOIN protocol_diversity proto ON tx.wallet = proto.wallet
    LEFT JOIN transfer_usd transfer ON tx.wallet = transfer.wallet
    LEFT JOIN interaction_diversity interact ON tx.wallet = interact.wallet
    LEFT JOIN activity_range act_range ON tx.wallet = act_range.wallet
    LEFT JOIN transfer_value_stats value_stats ON tx.wallet = value_stats.wallet
    LEFT JOIN wallet_intent_counts intent ON tx.wallet = intent.wallet
    LEFT JOIN bridge_behavior bridge ON tx.wallet = bridge.wallet
)

SELECT *
FROM wallet_features
WHERE
    tx_per_month >= 1
    AND total_transfer_usd > 0
    AND active_duration_days >= 30
;
