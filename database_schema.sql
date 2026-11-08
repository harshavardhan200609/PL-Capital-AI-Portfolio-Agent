-- Database schema with foreign key relationships and indices
-- This file should be executed to set up the database structure

-- Create tables in dependency order

CREATE TABLE IF NOT EXISTS sectors (
    sector_id INTEGER PRIMARY KEY,
    sector_name TEXT NOT NULL UNIQUE,
    sector_description TEXT,
    industry_group TEXT
);

CREATE TABLE IF NOT EXISTS securities (
    security_id INTEGER PRIMARY KEY,
    symbol TEXT NOT NULL UNIQUE,
    company_name TEXT NOT NULL,
    asset_type TEXT NOT NULL CHECK(asset_type IN ('Stock', 'Bond')),
    sector_id INTEGER,
    market_cap REAL,
    current_price REAL,
    currency TEXT DEFAULT 'USD',
    exchange TEXT,
    country TEXT,
    listing_date DATE,
    maturity_date DATE,  -- For bonds
    coupon_rate REAL,    -- For bonds
    FOREIGN KEY (sector_id) REFERENCES sectors(sector_id)
);

CREATE TABLE IF NOT EXISTS benchmarks (
    benchmark_id INTEGER PRIMARY KEY,
    benchmark_name TEXT NOT NULL UNIQUE,
    benchmark_symbol TEXT,
    benchmark_type TEXT,
    description TEXT,
    inception_date DATE
);

CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id INTEGER PRIMARY KEY,
    portfolio_name TEXT NOT NULL UNIQUE,
    creation_date DATE,
    target_risk_level TEXT,
    total_aum REAL,
    strategy_type TEXT,
    benchmark_index TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS holdings (
    holding_id INTEGER PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    security_id INTEGER NOT NULL,
    quantity REAL NOT NULL,
    purchase_price REAL,
    purchase_date DATE,
    current_weight REAL,
    cost_basis REAL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    FOREIGN KEY (security_id) REFERENCES securities(security_id),
    UNIQUE(portfolio_id, security_id)
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    security_id INTEGER NOT NULL,
    transaction_type TEXT CHECK(transaction_type IN ('BUY', 'SELL')) NOT NULL,
    quantity REAL NOT NULL,
    price REAL NOT NULL,
    transaction_date DATE NOT NULL,
    fees REAL DEFAULT 0,
    settlement_date DATE,
    notes TEXT,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    FOREIGN KEY (security_id) REFERENCES securities(security_id)
);

CREATE TABLE IF NOT EXISTS historical_prices (
    price_id INTEGER PRIMARY KEY,
    security_id INTEGER NOT NULL,
    price_date DATE NOT NULL,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL NOT NULL,
    volume INTEGER,
    adjusted_close REAL,
    FOREIGN KEY (security_id) REFERENCES securities(security_id),
    UNIQUE(security_id, price_date)
);

CREATE TABLE IF NOT EXISTS portfolio_performance (
    performance_id INTEGER PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    performance_date DATE NOT NULL,
    nav REAL NOT NULL,
    total_return_1m REAL,
    total_return_3m REAL,
    total_return_6m REAL,
    total_return_1y REAL,
    volatility REAL,
    sharpe_ratio REAL,
    max_drawdown REAL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    UNIQUE(portfolio_id, performance_date)
);

CREATE TABLE IF NOT EXISTS risk_metrics (
    risk_id INTEGER PRIMARY KEY,
    portfolio_id INTEGER NOT NULL,
    calculation_date DATE NOT NULL,
    var_95 REAL,
    var_99 REAL,
    cvar_95 REAL,
    beta REAL,
    correlation_sp500 REAL,
    tracking_error REAL,
    information_ratio REAL,
    sortino_ratio REAL,
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id),
    UNIQUE(portfolio_id, calculation_date)
);

-- Create indices for better query performance

-- Primary lookup indices
CREATE INDEX IF NOT EXISTS idx_securities_symbol ON securities(symbol);
CREATE INDEX IF NOT EXISTS idx_securities_sector ON securities(sector_id);
CREATE INDEX IF NOT EXISTS idx_securities_asset_type ON securities(asset_type);

-- Foreign key indices for joins
CREATE INDEX IF NOT EXISTS idx_holdings_portfolio ON holdings(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_holdings_security ON holdings(security_id);
CREATE INDEX IF NOT EXISTS idx_holdings_date ON holdings(purchase_date);
CREATE INDEX IF NOT EXISTS idx_transactions_portfolio ON transactions(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_transactions_security ON transactions(security_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_historical_prices_security ON historical_prices(security_id);
CREATE INDEX IF NOT EXISTS idx_historical_prices_date ON historical_prices(price_date);
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_portfolio ON portfolio_performance(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_portfolio_performance_date ON portfolio_performance(performance_date);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_portfolio ON risk_metrics(portfolio_id);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_date ON risk_metrics(calculation_date);

-- Composite indices for common query patterns
CREATE INDEX IF NOT EXISTS idx_holdings_portfolio_security ON holdings(portfolio_id, security_id);
CREATE INDEX IF NOT EXISTS idx_transactions_portfolio_date ON transactions(portfolio_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_historical_prices_security_date ON historical_prices(security_id, price_date);