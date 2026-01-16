#!/usr/bin/env python3
"""
Complete debug script to diagnose all bot issues in one place
"""

import sys
import os
import asyncio
sys.path.append('src')

from multi_chain_manager import MultiChainManager

async def complete_bot_debug():
    print("🚀 COMPLETE BOT DEBUG - ALL TESTS IN ONE")
    print("=" * 70)
    
    WALLET_ADDRESS = "0x49CF5f42914f8fa5746dAdC6fcb6E2E9EE1cd4E5"
    
    # ========================================
    # TEST 1: Network Configuration
    # ========================================
    print("📋 TEST 1: NETWORK CONFIGURATION")
    print("-" * 40)
    
    manager = MultiChainManager()
    print(f"📊 Total networks configured: {len(manager.networks)}")
    print(f"🌐 Networks found:")
    
    for network_name, network_config in manager.networks.items():
        print(f"   • {network_name}: {network_config.name} (Chain ID: {network_config.chain_id})")
    
    # ========================================
    # TEST 2: Network Connections
    # ========================================
    print(f"\n🔌 TEST 2: NETWORK CONNECTIONS")
    print("-" * 40)
    
    connection_results = await manager.connect_to_all_networks()
    connected_count = sum(connection_results.values())
    print(f"✅ Connected to {connected_count}/{len(manager.networks)} networks")
    
    connected_networks = []
    for network_name, connected in connection_results.items():
        status = "✅ CONNECTED" if connected else "❌ FAILED"
        network_display = manager.networks[network_name].name
        print(f"   {network_display}: {status}")
        if connected:
            connected_networks.append(network_name)
    
    if connected_count == 0:
        print("❌ CRITICAL: No networks connected - bot cannot work!")
        return
    
    # ========================================
    # TEST 3: Individual Balance Checks
    # ========================================
    print(f"\n💰 TEST 3: INDIVIDUAL BALANCE CHECKS")
    print("-" * 40)
    
    balances_found = {}
    for network_name in connected_networks:
        print(f"\n🔍 Testing {manager.networks[network_name].name}...")
        
        try:
            # Method 1: Direct balance check
            balance = await manager.check_native_balance(network_name, WALLET_ADDRESS)
            if balance is not None:
                network_display = manager.networks[network_name].name
                token_symbol = manager.networks[network_name].native_token_symbol
                print(f"   Direct check: {balance:.6f} {token_symbol}")
                balances_found[network_name] = balance
                
                if balance > 0:
                    print(f"   🎉 HAS FUNDS!")
                else:
                    print(f"   ⚠️  Empty wallet")
            else:
                print(f"   ❌ Direct balance check failed")
            
            # Method 2: Full balance info
            balance_info = await manager.get_balance_for_network(network_name, WALLET_ADDRESS)
            if balance_info:
                print(f"   Full check: {balance_info.native_balance:.6f} {manager.networks[network_name].native_token_symbol}")
                print(f"   USD Value: ${balance_info.native_balance_usd:.2f}")
            else:
                print(f"   ❌ Full balance check failed")
                
        except Exception as e:
            print(f"   ❌ Balance check error: {e}")
    
    # ========================================
    # TEST 4: Gas Estimation for Each Network
    # ========================================
    print(f"\n⛽ TEST 4: GAS ESTIMATION FOR EACH NETWORK")
    print("-" * 40)
    
    gas_estimates = {}
    test_trade_amount = 10.0
    
    for network_name in connected_networks:
        print(f"\n🔍 Testing gas for {manager.networks[network_name].name}...")
        
        try:
            gas_estimate = await manager.get_gas_estimate(network_name, test_trade_amount)
            
            if gas_estimate:
                print(f"   ✅ Gas estimation successful")
                print(f"   💰 Cost: ${gas_estimate.estimated_cost_usd:.3f}")
                print(f"   📊 Score: {gas_estimate.total_cost_score:.1f}%")
                print(f"   ⛽ Gas price: {gas_estimate.gas_price_gwei:.1f} gwei")
                print(f"   💎 Liquidity: {gas_estimate.liquidity_score}/100")
                gas_estimates[network_name] = gas_estimate
            else:
                print(f"   ❌ Gas estimation returned None")
                
        except Exception as e:
            print(f"   ❌ Gas estimation failed: {e}")
    
    # ========================================
    # TEST 5: Find Optimal Network
    # ========================================
    print(f"\n🎯 TEST 5: FIND OPTIMAL NETWORK")
    print("-" * 40)
    
    optimal_network = None
    gas_estimate = None
    
    try:
        optimal_network, gas_estimate = await manager.find_optimal_network(test_trade_amount)
        
        if optimal_network and gas_estimate:
            network_display = manager.networks[optimal_network].name
            print(f"✅ Optimal network found: {network_display}")
            print(f"💰 Gas cost: ${gas_estimate.estimated_cost_usd:.3f}")
            print(f"📊 Cost percentage: {gas_estimate.total_cost_score:.1f}%")
            print(f"🎉 SUCCESS: Bot should be able to trade!")
        elif optimal_network and not gas_estimate:
            network_display = manager.networks[optimal_network].name
            print(f"⚠️  Network found but no gas estimate: {network_display}")
            print(f"❌ This is likely causing the 'insufficient tokens' error")
        else:
            print(f"❌ No optimal network found for ${test_trade_amount} trade")
            print(f"❌ This is why your bot says 'insufficient native tokens'")
            
    except Exception as e:
        print(f"❌ find_optimal_network failed: {e}")
    
    # ========================================
    # TEST 6: Get All Balances (Bot Method)
    # ========================================
    print(f"\n📊 TEST 6: GET ALL BALANCES (BOT METHOD)")
    print("-" * 40)
    
    try:
        all_balances = await manager.get_all_balances(WALLET_ADDRESS)
        
        if all_balances:
            total_usd = sum(balance.native_balance_usd for balance in all_balances.values())
            print(f"✅ get_all_balances() successful")
            print(f"💰 Total Portfolio: ${total_usd:.2f}")
            print(f"📈 Networks with balances:")
            
            funded_networks = []
            for network_name, balance in all_balances.items():
                network_display = manager.networks[network_name].name
                token_symbol = manager.networks[network_name].native_token_symbol
                print(f"   {network_display}: {balance.native_balance:.6f} {token_symbol} (${balance.native_balance_usd:.2f})")
                
                if balance.native_balance > 0:
                    print(f"     🎉 FUNDED")
                    funded_networks.append(network_name)
                else:
                    print(f"     ⚠️  EMPTY")
            
            if funded_networks:
                print(f"\n✅ Bot should detect {len(funded_networks)} funded networks")
            else:
                print(f"\n❌ Bot thinks all networks are empty!")
                
        else:
            print(f"❌ get_all_balances() returned None")
            print(f"❌ This is why your bot can't see any balances!")
            
    except Exception as e:
        print(f"❌ get_all_balances() failed: {e}")
    
    # ========================================
    # SUMMARY & DIAGNOSIS
    # ========================================
    print(f"\n🏁 SUMMARY & DIAGNOSIS")
    print("=" * 70)
    
    print(f"📊 Networks configured: {len(manager.networks)}")
    print(f"🔌 Networks connected: {connected_count}")
    print(f"💰 Networks with funds: {len([n for n, b in balances_found.items() if b > 0])}")
    print(f"⛽ Networks with gas estimates: {len(gas_estimates)}")
    
    # Identify the specific issue
    if connected_count == 0:
        print(f"\n❌ ISSUE: No network connections")
        print(f"💡 FIX: Check RPC endpoints")
    elif len(balances_found) == 0:
        print(f"\n❌ ISSUE: Balance detection failing")
        print(f"💡 FIX: Check balance checking logic")
    elif len(gas_estimates) == 0:
        print(f"\n❌ ISSUE: Gas estimation failing")
        print(f"💡 FIX: Check gas estimation methods")
    elif not optimal_network:
        print(f"\n❌ ISSUE: Network selection failing")
        print(f"💡 FIX: Check find_optimal_network logic")
    else:
        print(f"\n✅ DIAGNOSIS: Everything should work!")
        print(f"🎉 Your bot should be able to trade")

if __name__ == "__main__":
    asyncio.run(complete_bot_debug())