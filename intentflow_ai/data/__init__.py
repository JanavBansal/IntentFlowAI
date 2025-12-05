"""Data ingestion and storage utilities."""

from intentflow_ai.data.ingestion import DataIngestionWorkflow
from intentflow_ai.data.sources import DataSource, SourceRegistry

__all__ = ["DataIngestionWorkflow", "DataSource", "SourceRegistry"]

# EODHD fundamental data provider
try:
    from intentflow_ai.data.eodhd_provider import (
        EODHDProvider,
        EODHDConfig,
        get_eodhd_provider,
    )
    __all__.extend([
        "EODHDProvider",
        "EODHDConfig",
        "get_eodhd_provider",
    ])
except ImportError:
    pass

# Corporate actions handler
try:
    from intentflow_ai.data.corporate_actions import (
        CorporateActionsHandler,
        CorporateActionsConfig,
        CorporateAction,
    )
    __all__.extend([
        "CorporateActionsHandler",
        "CorporateActionsConfig",
        "CorporateAction",
    ])
except ImportError:
    pass

# Macro data provider
try:
    from intentflow_ai.data.macro_provider import (
        MacroDataProvider,
        MacroConfig,
        get_macro_provider,
    )
    __all__.extend([
        "MacroDataProvider",
        "MacroConfig",
        "get_macro_provider",
    ])
except ImportError:
    pass

# NSE options data provider
try:
    from intentflow_ai.data.nse_options_provider import (
        NSEOptionsProvider,
        OptionsConfig,
    )
    __all__.extend([
        "NSEOptionsProvider",
        "OptionsConfig",
    ])
except ImportError:
    pass

# Liquidity filter
try:
    from intentflow_ai.data.filters.liquidity import (
        LiquidityFilter,
        LiquidityConfig,
    )
    __all__.extend([
        "LiquidityFilter",
        "LiquidityConfig",
    ])
except ImportError:
    pass
