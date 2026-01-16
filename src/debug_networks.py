#!/usr/bin/env python3
"""
Debug script to check what networks are actually configured
"""

import sys
import os
sys.path.append('src')

from multi_chain_manager import MultiChainManager

def debug_networks():
    print("🔍 DEBUGGING NETWORK CONFIGURATION")
    print("=" * 50)
    
    # Create manager
    manager = MultiChainManager()
    
    # Check configured networks
    print(f"📊 Total networks configured: {len(manager.networks)}")
    print(f"🌐 Networks found:")
    
    for network_name, network_config in manager.networks.items():
        print(f"   • {network_name}: {network_config.name} (Chain ID: {network_config.chain_id})")
    
    print("\n✅ This should only show Polygon, Optimism, and Base")
    print("❌ If you see Ethereum or Arbitrum, they weren't commented out properly")

if __name__ == "__main__":
    debug_networks()