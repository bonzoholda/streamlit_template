from typing import Dict, List, Any, Optional
import json
import os
import logging

class Config:
    """
    Configuration manager for the trading bot.
    
    Responsibilities:
    1. Loading and saving configuration from/to file
    2. Providing access to configuration parameters
    3. Validating configuration changes
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self.load_config()
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> Dict:
        """
        Load configuration from file or create default.
        
        Returns:
            Configuration dictionary
        """
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading config file: {str(e)}")
                return self._get_default_config()
        else:
            # Create default config
            default_config = self._get_default_config()
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config: Dict = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save (uses current if None)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            if config:
                self.config = config
                
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving config file: {str(e)}")
            return False
    
    def update_config(self, new_config: Dict) -> bool:
        """
        Update configuration with new values.
        
        Args:
            new_config: New configuration dictionary
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Merge new config with existing
            self._merge_config(self.config, new_config)
            
            # Validate the updated config
            if not self._validate_config(self.config):
                self.logger.error("Invalid configuration")
                return False
                
            # Save the updated config
            return self.save_config()
            
        except Exception as e:
            self.logger.error(f"Error updating config: {str(e)}")
            return False
    
    def get_config(self) -> Dict:
        """
        Get the entire configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config
    
    def get_exchange_config(self) -> Dict:
        """
        Get exchange-specific configuration.
        
        Returns:
            Exchange configuration dictionary
        """
        return self.config.get("exchange", {})
    
    def get_trading_config(self) -> Dict:
        """
        Get trading-specific configuration.
        
        Returns:
            Trading configuration dictionary
        """
        return self.config.get("trading", {})
    
    def get_risk_config(self) -> Dict:
        """
        Get risk management configuration.
        
        Returns:
            Risk management configuration dictionary
        """
        return self.config.get("risk", {})
    
    def get_compound_config(self) -> Dict:
        """
        Get profit compounding configuration.
        
        Returns:
            Profit compounding configuration dictionary
        """
        return self.config.get("compound", {})
    
    def get_strategy_parameters(self) -> Dict:
        """
        Get strategy-specific parameters based on selected strategy.
        
        Returns:
            Strategy parameters dictionary
        """
        strategy_name = self.config.get("trading", {}).get("strategy")
        
        if strategy_name == "GridTrading":
            return self.config.get("grid", {})
        elif strategy_name == "TrendFollowing":
            return self.config.get("trend", {})
        else:
            return {}
    
    def get_trading_symbol(self) -> str:
        """
        Get the configured trading symbol.
        
        Returns:
            Trading symbol string (e.g., "BTC-USDT")
        """
        return self.config.get("trading", {}).get("symbol", "BTC-USDT")
    
    def get_trading_interval(self) -> str:
        """
        Get the configured trading interval.
        
        Returns:
            Trading interval string (e.g., "15m")
        """
        return self.config.get("trading", {}).get("interval", "15m")
    
    def is_test_mode(self) -> bool:
        """
        Check if test mode is enabled.
        
        Returns:
            True if test mode is enabled, False otherwise
        """
        return self.config.get("exchange", {}).get("test_mode", True)
    
    def is_compounding_enabled(self) -> bool:
        """
        Check if profit compounding is enabled.
        
        Returns:
            True if compounding is enabled, False otherwise
        """
        return self.config.get("compound", {}).get("enabled", True)
    
    def get_risk_parameters(self) -> Dict:
        """
        Get risk management parameters.
        
        Returns:
            Dictionary of risk management parameters
        """
        risk_config = self.config.get("risk", {})
        
        return {
            "max_position_size": risk_config.get("max_position_size_percent", 10.0),
            "max_drawdown": risk_config.get("max_drawdown_percent", 15.0),
            "daily_risk_limit": risk_config.get("max_daily_risk_percent", 5.0)
        }
    
    def _get_default_config(self) -> Dict:
        """
        Create a default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "exchange": {
                "api_key": "",
                "api_secret": "",
                "passphrase": "",
                "test_mode": True
            },
            "trading": {
                "symbol": "BTC-USDT",
                "strategy": "TrendFollowing",
                "interval": "15m",
                "max_open_positions": 3
            },
            "risk": {
                "max_position_size_percent": 10,
                "stop_loss_percent": 2,
                "take_profit_percent": 4, 
                "max_daily_risk_percent": 5,
                "max_drawdown_percent": 15
            },
            "compound": {
                "enabled": True,
                "target_compound_rate": 100,  # 100% (reinvest all profits)
                "compound_frequency": "Daily"
            },
            "trend": {
                "short_ema": 9,
                "long_ema": 21
            },
            "grid": {
                "upper_limit": 0,
                "lower_limit": 0,
                "grid_levels": 5
            }
        }
    
    def _validate_config(self, config: Dict) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required sections
        required_sections = ["exchange", "trading", "risk", "compound"]
        for section in required_sections:
            if section not in config:
                self.logger.error(f"Missing required config section: {section}")
                return False
                
        # Validate trading parameters
        trading = config.get("trading", {})
        if not trading.get("symbol"):
            self.logger.error("Missing trading symbol")
            return False
            
        if trading.get("strategy") not in ["TrendFollowing", "GridTrading"]:
            self.logger.error(f"Invalid trading strategy: {trading.get('strategy')}")
            return False
            
        if trading.get("interval") not in ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]:
            self.logger.error(f"Invalid trading interval: {trading.get('interval')}")
            return False
        
        # Validate risk parameters
        risk = config.get("risk", {})
        for param in ["max_position_size_percent", "stop_loss_percent", "take_profit_percent", 
                     "max_daily_risk_percent", "max_drawdown_percent"]:
            if param not in risk:
                self.logger.error(f"Missing risk parameter: {param}")
                return False
            if not isinstance(risk[param], (int, float)) or risk[param] <= 0:
                self.logger.error(f"Invalid risk parameter value: {param}={risk[param]}")
                return False
        
        # Validate compounding parameters
        compound = config.get("compound", {})
        if "enabled" not in compound:
            self.logger.error("Missing compound.enabled parameter")
            return False
        if compound.get("enabled") and "target_compound_rate" not in compound:
            self.logger.error("Missing target_compound_rate parameter")
            return False
        if compound.get("enabled") and compound.get("compound_frequency") not in ["Daily", "Weekly", "Monthly"]:
            self.logger.error(f"Invalid compound frequency: {compound.get('compound_frequency')}")
            return False
            
        return True
    
    def _merge_config(self, target: Dict, source: Dict) -> None:
        """
        Recursively merge source dictionary into target dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
