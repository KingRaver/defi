#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced meme_phrases.py
Contains expanded meme phrases and templates for crypto market commentary
with improved variety and categorization for diverse social media responses.
"""

import random
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple

# Market sentiment categories
class MarketSentiment(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILE = "volatile"
    RECOVERING = "recovering"
    EUPHORIC = "euphoric"
    CAPITULATION = "capitulation"
    UNCERTAIN = "uncertain"
    SIDEWAYS = "sideways"
    FOMO = "fomo"
    FEAR = "fear"

# Conversation tone categories
class ConversationTone(Enum):
    HUMOROUS = "humorous"
    ANALYTICAL = "analytical"
    SKEPTICAL = "skeptical"
    EDUCATIONAL = "educational"
    CASUAL = "casual"
    ENTHUSIASTIC = "enthusiastic"
    SARCASTIC = "sarcastic"
    PHILOSOPHICAL = "philosophical"
    CONTRARIAN = "contrarian"
    TECHNICAL = "technical"
    MEME = "meme"

# Audience types for tailored responses
class AudienceType(Enum):
    TRADER = "trader"
    DEVELOPER = "developer"
    INVESTOR = "investor"
    NEWBIE = "newbie"
    INFLUENCER = "influencer"
    PROJECT_TEAM = "project_team"
    RESEARCHER = "researcher"
    JOURNALIST = "journalist"
    ENTHUSIAST = "enthusiast"

# Expanded generic crypto meme phrases
MEME_PHRASES = {
    'bullish': [
        "Going to the moon!",
        "Diamond hands activated!",
        "Bears getting rekt!",
        "Pump it up!",
        "Green candles incoming!",
        "Bull market is back on the menu!",
        "Number go up technology at its finest!",
        "Up only mode engaged!",
        "Lambos incoming!",
        "This is gentlemen.",
        "WAGMI - We're all gonna make it!",
        "Strapping in for liftoff!",
        "The bulls are running!",
        "Pump up the volume!",
        "Rocket emojis intensify...",
        "Hide your shorts!"
    ],
    'bearish': [
        "This is fine. (It's not fine)",
        "Bear market things",
        "Buying the dip... again",
        "Pain.",
        "Liquidation cascade incoming",
        "Ramen back on the menu",
        "Guess I'll just hodl longer",
        "Capitulation incoming",
        "Exit liquidity secured",
        "Catching falling knives",
        "Blood in the streets",
        "Holding these bags is a workout",
        "At least we have the tech",
        "Generational buying opportunity (again)",
        "Time to zoom out... WAY out",
        "NGMI - Not gonna make it"
    ],
    'neutral': [
        "Boring market is boring",
        "Crab market continues",
        "Sideways action forever",
        "Waiting for volatility",
        "Consolidation phase",
        "Chop city",
        "Another day in crypto",
        "Range-bound trading continues",
        "Still early™",
        "Perfectly balanced, as all things should be",
        "Time to touch grass",
        "Market taking a breather",
        "Neither bull nor bear, just crab",
        "Accumulation phase intensifies",
        "Silence is deafening in the markets",
        "Is this thing on?"
    ],
    'volatile': [
        "Hold onto your hats!",
        "Traders' nightmare",
        "Epic volatility",
        "Rollercoaster mode activated",
        "Chop city",
        "Whiplash market in full effect",
        "Heart attack market continues",
        "Blood pressure rising with these swings",
        "Market can't make up its mind",
        "Ping-ponging between support and resistance",
        "Chart looking like an EKG",
        "Trading this is like riding a mechanical bull",
        "Market's had too much coffee",
        "Schrodinger's market: simultaneously bullish and bearish",
        "Exit liquidity, then entry liquidity, then exit again..."
    ],
    'recovering': [
        "Finding a bottom",
        "Green shoots appearing",
        "Relief rally time",
        "Bottom fishers rewarded",
        "Comeback season",
        "V-shaped recovery incoming?",
        "Bears sweating a little",
        "Is the worst over?",
        "Holding paid off",
        "Survival mode deactivated",
        "First signs of spring after crypto winter",
        "Relief rally or dead cat bounce?",
        "Recovery mode engaged",
        "Slow and steady wins the race",
        "Hope creeping back into the market"
    ],
    'euphoric': [
        "Nothing can stop this rally!",
        "Never selling, never surrender!",
        "Generational wealth being created!",
        "I'm never going to financially recover from... being too rich!",
        "Up only, forever!",
        "Euphoria levels through the roof!",
        "Retirement calculator checking intensifies",
        "What's a bear market?",
        "This time is different, obviously!",
        "Looking at real estate listings already",
        "Imagine not being in crypto right now!",
        "Financial freedom incoming!",
        "Telling the boss off rehearsal in progress",
        "New paradigm confirmed!",
        "Never have to work again!"
    ],
    'capitulation': [
        "I'm never financially recovering from this",
        "Deleting price apps and living in blissful ignorance",
        "Wake me up when it's over",
        "Unsubscribe from the pain",
        "Forced hodling engaged",
        "This isn't even my final form of pain",
        "The dip keeps dipping",
        "Tax loss harvesting championship",
        "Surrendering to the market gods",
        "Maybe my parents were right about crypto",
        "Calling it an 'unrealized loss' helps me sleep",
        "Is Wendy's hiring?",
        "From diamond hands to dust",
        "Gave up checking portfolio, focuses on mental health instead",
        "Maybe I should've invested in index funds"
    ],
    'uncertain': [
        "What even is this market?",
        "Confused trader noises",
        "Analysis paralysis incoming",
        "Market playing 4D chess",
        "Nobody knows anything",
        "Experts completely clueless",
        "The charts are speaking in riddles",
        "TA experts in shambles",
        "Direction unclear, try again later",
        "Market wearing a blindfold",
        "Even the whales are confused",
        "Waiting for a sign, any sign",
        "Trust the plan... if there is one",
        "Sentiment mixer on full blast",
        "I'll just flip a coin at this point"
    ],
    'fomo': [
        "Last chance to buy under $X!",
        "Don't miss this train!",
        "Everybody's making gains but me!",
        "Getting in now or regretting later?",
        "Sleep now, or buy now?",
        "This rocket is leaving the station!",
        "Your friends will be jealous you got in now",
        "Once-in-a-lifetime opportunity!",
        "Staying on the sidelines is the real risk",
        "Getting priced out in real-time",
        "If you missed the last one, here's your second chance",
        "Millionaires are being made RIGHT NOW",
        "The next wave is here and it's massive",
        "When you look back, this was the moment",
        "Imagine telling people you had the chance but didn't take it"
    ],
    'fear': [
        "Exit while you still can",
        "Abandon all hope",
        "Catching falling knives is dangerous",
        "Winter is coming",
        "Last one out, turn off the lights",
        "The end is near",
        "Don't try to catch this falling knife",
        "This isn't a dip, it's a crater",
        "Institutions dumping on retail again",
        "They knew all along",
        "Sell now or forever hold your bags",
        "This isn't even the bottom yet",
        "Paradigm shift: downwards only from here",
        "The experiment has failed",
        "Game over, insert more coins"
    ]
}

# Time context phrases - for different market timeframes
TIME_CONTEXT_PHRASES = {
    'short_term': [
        "The 5-minute chart is all that matters!",
        "Day trading this volatility is for the brave.",
        "Scalpers paradise right now!",
        "Hourly RSI looking spicy!",
        "The 1-hour close will be critical.",
        "Intraday traders having a field day!",
        "5-minute candles giving me whiplash.",
        "Short-term traders, this is your moment.",
        "Watching these minute candles like a hawk.",
        "Scalp city population: everyone.",
        "This hourly volume is insane!",
        "Minute-to-minute action is intense today."
    ],
    'mid_term': [
        "Weekly chart looking interesting.",
        "The daily trend is clear as day.",
        "Swing traders are eating well this month.",
        "This weekly candle could change everything.",
        "Daily RSI showing potential reversals.",
        "Monthly close will be telling.",
        "This swing trade opportunity is prime.",
        "Daily supports holding strong.",
        "Mid-term trend still intact despite noise.",
        "Weekly momentum building nicely.",
        "This daily pattern is textbook.",
        "Setting up for a great month ahead."
    ],
    'long_term': [
        "Zoom out and relax.",
        "The yearly chart is all you need to see.",
        "Long-term trend is your friend.",
        "Still early in the adoption curve.",
        "Reminder: 4-year cycles are still a thing.",
        "For the patient, generational wealth awaits.",
        "Think in years, not days.",
        "The macro view remains incredibly bullish.",
        "Long-term hodlers sleep better at night.",
        "This is a marathon, not a sprint.",
        "Yearly open to close still impressive.",
        "In the long run, fundamentals always win."
    ],
    'historical': [
        "Just like the 2017 run-up all over again.",
        "Reminds me of the 2013 pattern.",
        "We've seen this movie before in 2020.",
        "History doesn't repeat, but it rhymes.",
        "Same FUD, different cycle.",
        "Tale as old as crypto time.",
        "If you survived 2018, this is nothing.",
        "Remember how 2021 felt? We're getting there.",
        "The charts from 2016 look eerily similar.",
        "Learning from previous cycles is key.",
        "This selloff is nowhere near 2018 levels.",
        "Those who forget crypto history are doomed to repeat it."
    ]
}

# Market cycle phrases - to reference different market cycle phases
MARKET_CYCLE_PHRASES = {
    'accumulation': [
        "Smart money accumulating quietly.",
        "Whales are loading their bags.",
        "Stealth accumulation phase in progress.",
        "OGs buying while there's blood in the streets.",
        "Institutions accumulating under the radar.",
        "Wyckoff accumulation pattern clear as day.",
        "The real game is being played in silence now.",
        "Those buying now will be rewarded later.",
        "Quiet accumulation before the storm.",
        "Weak hands to diamond hands transfer in progress.",
        "This is where generational wealth begins.",
        "Buying when others are fearful."
    ],
    'markup': [
        "First stage of the bull run detected.",
        "Early bulls being rewarded.",
        "The markup phase has begun!",
        "Price discovery mode activated.",
        "First stage rockets igniting.",
        "Smart money in profit, retail starting to notice.",
        "The trend is your friend, and it's heading up.",
        "Markup phase looking healthy so far.",
        "This is what the start of a rally looks like.",
        "Momentum building, retail slowly waking up.",
        "Textbook markup phase, enjoy the ride.",
        "Higher highs and higher lows - the good stuff."
    ],
    'distribution': [
        "Whales are starting to distribute.",
        "Smart money slowly exiting positions.",
        "Subtle distribution happening while retail FOMOs in.",
        "Late stage euphoria detected.",
        "Institutional selling into retail FOMO.",
        "Classic distribution pattern forming.",
        "Watch for whales taking profits here.",
        "The smart money is already rotating out.",
        "Distribution disguised as consolidation.",
        "When your Uber driver gives crypto tips, it's distribution time.",
        "Selling pressure from early investors detected.",
        "This range is where bag transfers happen."
    ],
    'markdown': [
        "The markdown phase is upon us.",
        "The hangover after the party.",
        "Gravity is real in crypto too.",
        "What goes up must come down.",
        "Forced selling cascade in progress.",
        "The markdown will create opportunities for patient players.",
        "Retail exit liquidity almost exhausted.",
        "Healthy markdown to build the next cycle.",
        "This is where diamonds are formed under pressure.",
        "Markdown separates the tourists from the residents.",
        "Price finding its real support levels now.",
        "The necessary evil of every market cycle."
    ]
}

# Market psychology phrases - based on psychological market states
MARKET_PSYCHOLOGY_PHRASES = {
    'denial': [
        "It's just a temporary dip!",
        "This pullback is healthy!",
        "We'll be back at all-time highs next week!",
        "Just a quick correction before the next leg up!",
        "The fundamentals haven't changed!",
        "This is perfectly normal market behavior!",
        "Weak hands are being shaken out, that's all!",
        "Just market manipulation before the next pump!",
        "It's just testing support before continuing up!",
        "Nothing to worry about, just some profit-taking!"
    ],
    'panic': [
        "Everything is crashing! Save yourselves!",
        "I'm never going to financially recover from this!",
        "It's all over! Crypto is dead (again)!",
        "Abandon ship! This is not a drill!",
        "Complete market meltdown in progress!",
        "The bubble has officially burst!",
        "The end is here! Sell everything!",
        "We're going to zero! This time for real!",
        "This is the crypto apocalypse!",
        "There's no bottom to this crash!"
    ],
    'capitulation': [
        "I give up. Selling everything for whatever I can get.",
        "It's not worth it anymore. I'm out.",
        "Deleting all crypto apps and walking away.",
        "Taking my losses and never looking back.",
        "This experiment has failed. I surrender.",
        "From diamond hands to dust.",
        "Calling it quits after this last dump.",
        "The dream is over. Time to wake up.",
        "The pain isn't worth the potential gain anymore.",
        "Converting all crypto to stable and moving on with life."
    ],
    'disbelief': [
        "This bounce won't last.",
        "Dead cat bounce, don't be fooled.",
        "Bull trap in progress.",
        "Don't trust this rally, it's fake.",
        "This recovery is just exit liquidity for whales.",
        "The real drop hasn't even started yet.",
        "Just another fake-out before the real crash.",
        "Too early to celebrate, this won't hold.",
        "It's a relief rally, not a reversal.",
        "I'll believe it when we break the downtrend line."
    ],
    'optimism': [
        "The bottom is in!",
        "Recovery looking strong!",
        "Sentiment shifting to positive!",
        "Green shoots are appearing everywhere!",
        "The worst is behind us now!",
        "This uptrend looks sustainable!",
        "Fundamentals finally reflected in price!",
        "The new bull market has begun!",
        "Promising signs all across the market!",
        "The comeback story is just getting started!"
    ],
    'belief': [
        "The uptrend is confirmed!",
        "Higher highs and higher lows - it's real!",
        "This bull market is just warming up!",
        "The fundamentals are stronger than ever!",
        "We're seeing real adoption driving this rally!",
        "The trend reversal is now undeniable!",
        "Institutional money is flowing in for real!",
        "This isn't just a rally, it's a market shift!",
        "Conviction in the uptrend growing stronger!",
        "This time the rise is backed by real usage!"
    ],
    'thrill': [
        "We're making new all-time highs daily!",
        "The gains are absolutely insane right now!",
        "Every coin I touch turns to gold!",
        "This market is simply unstoppable!",
        "I can't believe these returns are real!",
        "The adrenaline from checking my portfolio is addictive!",
        "Never seen gains like this before!",
        "Every day is like Christmas in this market!",
        "The rush of these green candles is unbelievable!",
        "I've never felt so alive trading crypto!"
    ],
    'euphoria': [
        "We're all going to be millionaires!",
        "This time it's different, it will never crash again!",
        "The old rules don't apply anymore!",
        "Number will only go up forever now!",
        "Traditional finance is dead, crypto has won!",
        "I'm never selling, not even a satoshi!",
        "My altcoin will 1000x from here minimum!",
        "Generational wealth opportunity happening RIGHT NOW!",
        "We've entered a new paradigm of perpetual growth!",
        "I might quit my job tomorrow, crypto trading is easier!"
    ],
    'complacency': [
        "Small dips are just opportunities to add more.",
        "The bull market will continue for years.",
        "Every correction is just a blip in the grand scheme.",
        "No need to take profits yet, much higher to go.",
        "The market is so strong, nothing can stop it now.",
        "Dips are transitory in this market environment.",
        "No need to worry about a crash, the market is resilient.",
        "We've established a new floor, it won't go lower.",
        "Risk management? In this economy?",
        "This level of growth is the new normal."
    ]
}

# Token-specific meme phrases templates
# {token} will be replaced with the actual token name
TOKEN_MEME_PHRASES = {
    'bullish': [
        "{token} taking off like a rocket!",
        "{token} bulls feasting today!",
        "Smart money loading up on {token}!",
        "{token} showing massive strength!",
        "{token} breaking through resistance like it's nothing!",
        "{token} whales accumulating hard!",
        "Imagine not having {token} in your portfolio right now!",
        "{token} outperforming everything in sight!",
        "{token}'s price action looking absolutely spectacular!",
        "The {token} train has left the station!",
        "If you're not watching {token} right now, you're missing out!",
        "{token} making millionaires as we speak!",
        "This {token} breakout is just getting started!",
        "{token} flexing on the entire market right now!",
        "{token} holders eating good tonight!",
        "{token} leading the market like a boss!"
    ],
    'bearish': [
        "{token} taking a breather after its epic run",
        "{token} discount sale! Everything must go!",
        "{token} testing HODLer conviction",
        "Paper hands shaking out of {token}",
        "{token} hitting support levels - time to buy?",
        "Weak hands folding on {token}",
        "{token} whales creating liquidity for a bigger move",
        "{token} bear trap in progress",
        "{token} dip bringing out the bargain hunters",
        "Is this the bottom for {token} or just the beginning?",
        "{token} falling knife - careful catching it!",
        "Even {token} can't escape market gravity",
        "This {token} correction was long overdue",
        "{token} finding out who the real believers are",
        "{token} price action making permabulls nervous",
        "{token} cooling off before the next leg up?"
    ],
    'neutral': [
        "{token} accumulation phase in progress",
        "{token} coiling for the next big move",
        "Smart money quietly accumulating {token}",
        "{token} volume drying up - calm before the storm?",
        "{token} trading in a tight range",
        "{token} consolidating after recent volatility",
        "Patience is key with {token} right now",
        "{token} building a solid base",
        "{token} in wait-and-see mode",
        "{token} taking a well-deserved break",
        "{token} price action boring but necessary",
        "{token} chart looking more stable than my relationship",
        "Watching {token} like paint drying right now",
        "{token} neither bullish nor bearish, just... there",
        "{token} in the eye of the market storm",
        "{token} putting everyone to sleep with this action"
    ],
    'volatile': [
        "{token} going absolutely crazy right now!",
        "{token} shorts and longs getting liquidated!",
        "{token} volatility through the roof!",
        "{token} making traders dizzy with these swings!",
        "{token} showing peak volatility!",
        "{token} traders need motion sickness pills!",
        "{token} chart looking like an EKG!",
        "{token} bouncing around like a pinball!",
        "{token} giving traders heart attacks today!",
        "{token} flash crashing then pumping in minutes!",
        "{token} price action is pure chaos theory!",
        "{token} trading not for the faint of heart today!",
        "{token} swings separating the pros from the amateurs!",
        "{token} playing ping pong between support and resistance!",
        "{token} volatility making day traders rich or broke!",
        "{token} chart looking like a roller coaster from hell!"
    ],
    'recovering': [
        "{token} showing signs of life!",
        "{token} recovery phase initiated!",
        "{token} bouncing back from the lows!",
        "{token} refusing to stay down!",
        "{token} resilience on display!",
        "{token} finding its footing after the dip!",
        "{token}'s recovery catching everyone by surprise!",
        "Dip buyers saving {token}!",
        "{token} bottoming process looking promising!",
        "{token} healing its chart wounds nicely!",
        "{token} proving the doubters wrong with this recovery!",
        "{token} holders finally seeing some relief!",
        "{token} showing why diamond hands are rewarded!",
        "{token} recovery mode activated, bears beware!",
        "{token} rising from the ashes impressively!",
        "{token} V-shaped recovery in progress!"
    ]
}

# Volume-specific phrase templates
VOLUME_PHRASES = {
    'significant_increase': [
        "{token} volume exploding! Something big brewing?",
        "Massive {token} volume spike detected!",
        "{token} volume through the roof - institutions loading up?",
        "Unprecedented {token} volume surge!",
        "{token} trading volume on steroids today!",
        "The {token} volume bars are going parabolic!",
        "Everyone's trading {token} today - volume off the charts!",
        "{token} seeing tsunami-level volume right now!",
        "Volume precedes price, and {token} volume is screaming!",
        "This {token} volume spike is no retail action - big players moving!"
    ],
    'moderate_increase': [
        "{token} volume picking up steam",
        "Growing interest in {token} with rising volumes",
        "{token} volume ticking up - early sign of momentum?",
        "Steady increase in {token} trading activity",
        "{token} volume starting to build",
        "More eyes on {token} as volume grows",
        "{token} seeing healthy volume expansion",
        "The {token} volume trend is encouraging",
        "Respectable volume growth for {token} today",
        "The {token} market is getting more liquid by the hour"
    ],
    'significant_decrease': [
        "{token} volume falling off a cliff",
        "{token} interest waning with plummeting volume",
        "{token} volume disappearing - traders moving elsewhere?",
        "Major drop in {token} trading activity",
        "{token} volume drought intensifying",
        "Everyone's forgotten about {token} - volume dead",
        "Ghost town vibes on the {token} trading pairs",
        "{token} volume declined to critical levels",
        "Traders abandoning {token} based on volume metrics",
        "The {token} liquidity is drying up fast"
    ],
    'moderate_decrease': [
        "{token} volume cooling off slightly",
        "Modest decline in {token} trading interest",
        "{token} volume easing back to normal levels",
        "Traders taking a break from {token} action",
        "{token} volume tapering down",
        "{token} seeing a natural volume pullback",
        "The {token} hype calming down based on volume",
        "{token} volume retreating from recent highs",
        "Slower day for {token} trading activity",
        "Trading interest in {token} taking a brief pause"
    ],
    'stable': [
        "{token} volume staying consistent",
        "Steady as she goes for {token} volume",
        "{token} trading at normal volume levels",
        "No major changes in {token} trading activity",
        "{token} volume in equilibrium",
        "Business as usual for {token} trading volume",
        "{token} showing predictable volume patterns",
        "Neither exciting nor concerning volume for {token}",
        "The {token} market functioning with routine volume",
        "Standard liquidity conditions for {token} today"
    ]
}

# Market comparison phrases
MARKET_COMPARISON_PHRASES = {
    'outperforming': [
        "{token} leaving the market in the dust!",
        "{token} outshining major crypto assets today!",
        "{token} flexing on the market!",
        "{token} showing the market how it's done!",
        "Market can't keep up with {token}'s pace!",
        "While the market struggles, {token} thrives!",
        "{token} in a league of its own performance-wise!",
        "The market is lagging behind {token}'s gains!",
        "{token} breaking away from the pack!",
        "If only the rest of the market performed like {token}!"
    ],
    'underperforming': [
        "{token} lagging behind market momentum",
        "{token} struggling while market pumps",
        "{token} needs to catch up to market performance",
        "Market strength overshadowing {token} today",
        "{token} taking a backseat to market gains",
        "The market rally leaving {token} behind",
        "{token} underperforming the broader crypto space",
        "Something holding {token} back while market soars",
        "{token} missing the memo about today's gains",
        "Market moving up, but {token} didn't get the invite"
    ],
    'correlated': [
        "{token} moving in lockstep with the market",
        "{token} and market correlation strengthening",
        "{token} riding the market wave",
        "Strong {token}-market correlation today",
        "{token} mirroring market price action",
        "{token} perfectly synced with market movements",
        "As goes the market, so goes {token}",
        "No divergence between {token} and broader market",
        "{token} and market correlation approaching 1.0",
        "The {token} chart is a carbon copy of the market today"
    ],
    'diverging': [
        "{token} breaking away from market correlation",
        "{token} charting its own path away from the market",
        "{token}-market correlation weakening",
        "{token} and market going separate ways",
        "{token} decoupling from market price action",
        "{token} ignoring the market's signals today",
        "Independence day for {token} price action",
        "{token} marching to the beat of its own drum",
        "Interesting divergence between {token} and the market",
        "The {token}-market correlation breaking down significantly"
    ]
}

# Smart money indicator phrases
SMART_MONEY_PHRASES = {
    'accumulating': [
        "Classic accumulation pattern forming on {token}",
        "Smart money quietly accumulating {token}",
        "Institutional accumulation detected on {token}",
        "Stealth accumulation phase underway for {token}",
        "Wyckoff accumulation signals on {token} chart",
        "Big wallets silently stacking {token}",
        "OTC {token} deals happening behind the scenes",
        "Under-the-radar {token} accumulation in progress",
        "The smart {token} positioning is happening now",
        "Subtle but unmistakable {token} accumulation signals"
    ],
    'distributing': [
        "Distribution pattern emerging on {token}",
        "Smart money distribution phase for {token}",
        "Institutional selling pressure on {token}",
        "{token} showing classic distribution signals",
        "Wyckoff distribution pattern on {token}",
        "Whales gradually offloading {token} positions",
        "The {token} distribution to retail is textbook",
        "Smart hands to weak hands {token} transfer happening",
        "Classic {token} distribution playing out on higher timeframes",
        "Early investors exiting {token} positions methodically"
    ],
    'divergence': [
        "Price-volume divergence on {token} - smart money move?",
        "{token} showing bullish divergence patterns",
        "Hidden divergence on {token} volume profile",
        "Smart money divergence signals flashing on {token}",
        "Institutional divergence pattern on {token}",
        "The {token} divergence is signaling a major move",
        "Textbook RSI divergence developing on {token}",
        "MACD divergence on {token} the smart money sees",
        "Multiple timeframe divergence for {token}",
        "Divergence masters looking closely at {token}"
    ],
    'abnormal_volume': [
        "Highly unusual volume pattern on {token}",
        "Abnormal {token} trading activity detected",
        "{token} volume anomaly spotted - insider action?",
        "Strange {token} volume signature today",
        "Statistically significant volume anomaly on {token}",
        "The {token} volume bars don't match the narrative",
        "Someone knows something about {token} - volume tells all",
        "Irregular {token} volume across exchanges",
        "Volume profile anomaly detected on {token}",
        "The {token} volume doesn't make sense unless..."
    ]
}

# Technical analysis specific phrases
TECHNICAL_ANALYSIS_PHRASES = {
    'support_resistance': [
        "{token} bouncing perfectly off key support!",
        "{token} struggling with overhead resistance",
        "{token} finding strong support at the 200-day MA",
        "Major resistance ahead for {token} at {level}",
        "Triple bottom support forming for {token}",
        "{token} needs to reclaim {level} to resume uptrend",
        "Critical support zone for {token} being tested now",
        "{token} resistance turned support - bullish signal",
        "Historic resistance level for {token} finally broken",
        "{token} in no man's land between support and resistance",
        "{token} establishing higher support levels",
        "Make or break support for {token} at {level}"
    ],
    'patterns': [
        "Beautiful cup and handle forming on {token}",
        "Classic head and shoulders pattern on {token}",
        "Bullish flag pattern developing for {token}",
        "{token} forming a textbook ascending triangle",
        "Double top pattern suggesting {token} reversal",
        "Inverse head and shoulders on {token} - bottoming?",
        "Falling wedge pattern on {token} suggests bullish reversal",
        "{token} chart showing perfect harmonic pattern",
        "Descending triangle on {token} - breakdown incoming?",
        "Three white soldiers pattern on {token} - bullish!",
        "Bearish engulfing pattern forming on {token}",
        "Bullish pennant on {token} suggesting continuation"
    ],
    'indicators': [
        "{token} RSI deeply oversold - bounce incoming?",
        "MACD crossover looking bullish for {token}",
        "Golden cross forming on {token} daily chart",
        "Death cross on {token} - proceed with caution",
        "{token} Bollinger Bands squeezing - volatility incoming",
        "Bullish divergence on {token} RSI",
        "{token} OBV rising while price falls - accumulation?",
        "Stochastic RSI showing overbought conditions for {token}",
        "Ichimoku cloud support holding strong for {token}",
        "{token} moving above the 200 EMA - long term bullish",
        "TD Sequential 9 count on {token} - reversal signal?",
        "Fibonacci retracement levels holding perfectly for {token}"
    ],
    'trend_analysis': [
        "{token} in clear uptrend with higher highs and higher lows",
        "Downtrend for {token} gaining momentum",
        "{token} broke downtrend line - potential reversal",
        "Perfect trendline respect from {token}",
        "{token} consolidating within uptrend channel",
        "Lower highs forming on {token} - trend weakening?",
        "{token} momentum slowing according to trend indicators",
        "Parabolic trend forming on {token} - unsustainable?",
        "Market structure broken for {token} - trend change likely",
        "{token} trading range tightening within trend",
        "Multiple timeframe trend alignment for {token} - strong signal",
        "{token} showing perfect trend pullback entry opportunity"
    ]
}

# DeFi specific phrases
DEFI_PHRASES = {
    'yield_farming': [
        "{token} yield farmers eating good tonight!",
        "The {token} farms are printing like crazy!",
        "APY on {token} farms looking juicy!",
        "Yield chasers flocking to {token} pools",
        "Sustainable yields or yield trap on {token}?",
        "{token} TVL growing as yields attract liquidity",
        "Impermanent loss warriors battling in {token} pools",
        "Yield optimization strategies for {token} trending",
        "The {token} farm rewards getting harvested hard",
        "Compounding {token} yields is the big brain move"
    ],
    'liquidity': [
        "{token} liquidity pools growing rapidly",
        "Thin {token} liquidity causing major slippage",
        "Whale providing massive {token} liquidity - bullish!",
        "Liquidity migrating from other projects to {token}",
        "{token} with incredibly deep order books now",
        "Concerning liquidity trends for {token} - proceed with caution",
        "{token} liquidity providers getting great returns",
        "Healthy two-sided liquidity developing for {token}",
        "Liquidity crisis brewing for {token}?",
        "The {token} liquidity flywheel in full effect"
    ],
    'governance': [
        "Critical {token} governance vote in progress",
        "The {token} DAO making power moves",
        "Governance attack concerns rising for {token}",
        "{token} holders voting on major protocol changes",
        "Fascinating governance experiment playing out with {token}",
        "Voter apathy affecting {token} governance decisions",
        "Progressive decentralization roadmap for {token} governance",
        "{token} whales controlling governance outcomes?",
        "Governance participation rewards boosting {token} voting",
        "{token} proposal voting closing soon - outcome unclear"
    ],
    'tokenomics': [
        "{token} tokenomics designed for sustainable growth",
        "Questionable {token} emission schedule",
        "Deflationary mechanisms putting pressure on {token} supply",
        "{token} burn rate accelerating",
        "The {token} supply shock thesis playing out",
        "Concerns about {token} token distribution",
        "Vesting schedule releasing more {token} - selling pressure?",
        "Balanced {token} tokenomics attracting long-term holders",
        "Game theory optimized {token} incentive structure",
        "Token velocity issues plaguing {token} price action"
    ]
}

# NFT specific phrases
NFT_PHRASES = {
    'collections': [
        "Floor price for {token} NFTs going through the roof!",
        "Rare {token} NFT just sold for record price!",
        "Volume on {token} NFT trading exploding!",
        "New {token} NFT collection dropping soon",
        "Unprecedented demand for {token} NFTs",
        "Whales accumulating blue chip {token} NFTs",
        "The {token} NFT roadmap causing excitement",
        "Floor sweeps happening on {token} collection",
        "Derivative collections boosting {token} NFT ecosystem",
        "The {token} NFT rarity tools showing hidden gems"
    ],
    'utility': [
        "{token} NFTs offering real utility",
        "Staking rewards for {token} NFT holders looking attractive",
        "New utility announcement pumping {token} NFT prices",
        "The {token} NFT ecosystem expanding with real use cases",
        "Passive income opportunities for {token} NFT holders",
        "Access token functionality driving {token} NFT demand",
        "Gaming integration boosting {token} NFT utility",
        "Membership benefits for {token} NFT owners expanding",
        "Utility vs. speculation debate raging in {token} community",
        "The {token} NFT utility roadmap looking promising"
    ],
    'market': [
        "NFT market conditions affecting {token} collection value",
        "Wash trading concerns in {token} NFT market metrics",
        "The {token} NFT market showing signs of recovery",
        "Volume profile for {token} NFTs showing accumulation",
        "Price discovery phase for new {token} NFT project",
        "Liquidity drying up in secondary {token} NFT markets",
        "Trading velocity increasing for {token} NFTs",
        "NFT market cycles affecting {token} valuations",
        "Correlation between {token} price and its NFTs breaking down",
        "Market sentiment shift impacting {token} NFT valuations"
    ]
}

# Layer 1 specific phrases
LAYER1_PHRASES = {
    'scaling': [
        "{token} TPS numbers looking impressive lately",
        "Scaling solution for {token} addressing bottlenecks",
        "Transaction speed on {token} outpacing competitors",
        "Network congestion on {token} despite scaling efforts",
        "The {token} roadmap focused on exponential scaling",
        "Sharding implementation boosting {token} performance",
        "Layer 2 synergy improving overall {token} ecosystem scaling",
        "Scaling vs. decentralization tradeoffs for {token}",
        "Horizontal scaling approach working well for {token}",
        "Vertical scaling limits becoming apparent for {token}"
    ],
    'security': [
        "{token} security model proving robust",
        "Audit results bolstering confidence in {token}",
        "Security concerns emerging around {token} implementation",
        "The {token} network hashrate reaching all-time highs",
        "Validator distribution improving {token} security posture",
        "Attack vectors for {token} being theoretically analyzed",
        "Economic security model for {token} showing resilience",
        "Bug bounty program for {token} attracting white hats",
        "Formal verification giving {token} security edge",
        "Security through obscurity criticism affecting {token}"
    ],
    'adoption': [
        "{token} adoption metrics growing exponentially",
        "Enterprise adoption of {token} gaining momentum",
        "Retail onboarding to {token} ecosystem simplified",
        "The {token} developer count trending upward",
        "Transaction count on {token} showing organic growth",
        "User experience improvements driving {token} adoption",
        "Institutional interest in {token} infrastructure increasing",
        "Regional adoption patterns emerging for {token}",
        "Adoption S-curve for {token} at early inflection point",
        "Mainstream adoption barriers being addressed by {token} team"
    ],
    'ecosystem': [
        "The {token} ecosystem expanding rapidly",
        "DApp diversity on {token} creating strong network effects",
        "Funding flowing into {token} ecosystem projects",
        "Developer tools for {token} reaching maturity",
        "The {token} ecosystem TVL distribution becoming healthier",
        "Cross-chain compatibility expanding {token} ecosystem reach",
        "Vertical integration within {token} ecosystem projects",
        "Composability advantages giving {token} ecosystem edge",
        "Ecosystem grants attracting talent to {token} development",
        "The {token} ecosystem narrative strengthening"
    ]
}

# Meme culture phrases
MEME_CULTURE_PHRASES = {
    'classic_memes': [
        "When lambo? {token} might have the answer!",
        "HODL {token} with those diamond hands!",
        "This is gentlemen for {token} holders!",
        "Funds are safu in {token} smart contracts",
        "The {token} FUD is strong today",
        "Zoom out on {token} chart to feel better",
        "Buy the dip on {token} activated",
        "WAGMI if you're holding {token}",
        "Probably nothing happening with {token} right now",
        "Few understand the true potential of {token}"
    ],
    'current_memes': [
        "{token} chart looking like an up only meme",
        "Galaxy brain move: accumulating {token} now",
        "Gigachad {token} accumulator vs. virgin panic seller",
        "The {token} community touching grass? Never!",
        "No cap, {token} looking bullish fr fr",
        "Dank {token} memes hitting different today",
        "It's giving financial freedom vibes for {token} holders",
        "The {token} thesis is bussin' bussin'",
        "Main character energy coming from {token} price action",
        "Chad {token} investors vs. soy altcoin bagholders"
    ],
    'trader_humor': [
        "My {token} trading strategy: buy high, sell low",
        "Trading {token} with leverage? I too like to live dangerously",
        "Just went all-in on {token}, what could possibly go wrong?",
        "The {token} 1-minute chart is all you need for financial advice",
        "Drawing lines on {token} charts to convince myself I know something",
        "I'm in it for the tech... the tech that makes {token} number go up",
        "My {token} position is underwater but so is my house",
        "Checking {token} price every 3 minutes is normal investor behavior",
        "Trading {token} has turned my hair grey, but my portfolio red",
        "Bought {token} at the top, now I'm a long-term investor by force"
    ],
    'community_jokes': [
        "The {token} community coping mechanism: high-grade hopium",
        "Average {token} enjoyer vs. typical fiat fan",
        "{token} maxis explaining why everything else is a scam",
        "Trust me bro, my {token} analysis is backed by vibes",
        "The {token} whitepaper readers vs. logo investors",
        "Explaining {token} tokenomics to my confused family",
        "Wife changing money coming from {token} investments",
        "The {token} community's financial advice: just trust the rainbow chart",
        "Traditional investors trying to understand why {token} just pumped 20%",
        "Explaining to IRS that my {token} losses were actually strategic"
    ]
}

# Regulatory and news phrases
REGULATORY_NEWS_PHRASES = {
    'regulation': [
        "Regulatory clarity for {token} emerging in key jurisdictions",
        "Regulatory FUD affecting {token} sentiment",
        "The {token} team proactively engaging with regulators",
        "Compliance measures being implemented for {token} protocol",
        "Regulatory arbitrage benefits for {token} disappearing",
        "Decentralization metrics for {token} addressing regulatory concerns",
        "The {token} foundation relocating for regulatory advantages",
        "Regulatory crackdown impacting {token} exchange listings",
        "Compliance pathway for {token} becoming clearer",
        "Legal opinion on {token} classification looking favorable"
    ],
    'adoption_news': [
        "Major enterprise adopting {token} for core infrastructure",
        "Institutional accumulation of {token} reported by analysts",
        "Partnership announcement boosting {token} visibility",
        "The {token} adoption by mainstream brand creating buzz",
        "Country-level adoption rumored for {token}",
        "Integration of {token} with existing financial rails",
        "Developer adoption metrics for {token} outpacing competitors",
        "Key opinion leaders endorsing {token} technology",
        "The {token} retail adoption curve accelerating",
        "Wallet adoption numbers for {token} growing steadily"
    ],
    'tech_developments': [
        "Major {token} upgrade implemented successfully",
        "The {token} roadmap milestone reached ahead of schedule",
        "Technical breakthrough for {token} scaling announced",
        "Developer activity on {token} repositories surging",
        "Protocol improvement proposal for {token} gaining traction",
        "Backwards compatibility concerns for {token} upgrade addressed",
        "The {token} testnet metrics exceeding expectations",
        "Critical bug in {token} protocol patched quietly",
        "Major refactoring of {token} codebase completed",
        "New technical whitepaper released detailing {token} innovations"
    ],
    'macro_factors': [
        "Macro economic conditions creating tailwinds for {token}",
        "Inflation hedge narrative strengthening for {token}",
        "The {token} correlation with traditional markets shifting",
        "Risk-on environment benefiting {token} asset class",
        "Monetary policy impacts on {token} liquidity becoming apparent",
        "Flight to quality within crypto benefiting {token}",
        "Global uncertainty driving safe haven demand for {token}",
        "Interest rate effects rippling into {token} markets",
        "The {token} narrative fitting current macro regime",
        "Structural shift in macro landscape favoring {token} thesis"
    ]
}

# Sentiment amplifiers - to make phrases more expressive
SENTIMENT_AMPLIFIERS = {
    'bullish': [
        "incredibly", "massively", "extremely", "undeniably",
        "ridiculously", "insanely", "enormously", "spectacularly",
        "phenomenally", "hugely", "intensely", "extraordinarily"
    ],
    'bearish': [
        "deeply", "severely", "profoundly", "intensely",
        "extremely", "seriously", "gravely", "dramatically",
        "disturbingly", "worryingly", "alarmingly", "critically"
    ],
    'neutral': [
        "mildly", "somewhat", "rather", "fairly",
        "reasonably", "moderately", "relatively", "marginally",
        "slightly", "nominally", "partially", "fractionally"
    ],
    'uncertain': [
        "possibly", "potentially", "perhaps", "maybe",
        "conceivably", "plausibly", "allegedly", "supposedly",
        "seemingly", "apparently", "ostensibly", "questionably"
    ],
    'intensity': [
        "absolutely", "completely", "totally", "utterly",
        "entirely", "wholly", "thoroughly", "fully",
        "fundamentally", "categorically", "decidedly", "definitively"
    ]
}

# Phrase templates for different audience types
AUDIENCE_TEMPLATES = {
    'trader': {
        'casual': "{token} looking ready for a breakout soon. Risk/reward is attractive here.",
        'technical': "Key support for {token} at {level} with RSI divergence suggesting potential reversal.",
        'bullish': "Bullish setup forming on {token} - upside target of {target} with stop below {support}.",
        'bearish': "Concerning signs for {token} with distribution pattern and weakening momentum.",
        'neutral': "Sideways action for {token} likely to continue between {range_low} and {range_high}."
    },
    'developer': {
        'casual': "The {token} codebase showing impressive improvements in recent commits.",
        'technical': "Latest {token} protocol upgrade demonstrates significant TPS improvements while maintaining security.",
        'bullish': "Developer activity on {token} repositories at all-time highs - always a bullish signal.",
        'bearish': "Concerning security vulnerabilities emerging in {token} implementation - team response critical.",
        'neutral': "The {token} development roadmap progressing at steady pace, focusing on core functionality."
    },
    'investor': {
        'casual': "Long-term value proposition for {token} remains strong despite short-term volatility.",
        'technical': "Fundamentals for {token} improving with rising adoption metrics and decreasing token velocity.",
        'bullish': "Institutional interest in {token} growing rapidly, similar to early Bitcoin adoption curve.",
        'bearish': "Token economics for {token} creating persistent selling pressure that needs addressing.",
        'neutral': "Risk-adjusted returns for {token} remain in line with broader crypto market."
    },
    'newbie': {
        'casual': "{token} is one of the interesting projects worth learning about in the crypto space.",
        'technical': "When looking at {token}, focus on the long-term utility and team rather than daily price.",
        'bullish': "Many see potential in {token} because of its strong community and real-world applications.",
        'bearish': "Always do your own research on projects like {token} and be cautious of online hype.",
        'neutral': "Projects like {token} show both the potential and risks of crypto investments."
    },
    'influencer': {
        'casual': "Been watching {token} closely lately - interesting developments worth sharing.",
        'technical': "My analysis shows {token} potentially setting up for significant move - details in thread.",
        'bullish': "Extremely bullish on {token} fundamentals right now - thread on why below.",
        'bearish': "Raising concerns about {token} that my followers should be aware of - transparency matters.",
        'neutral': "Balanced take on {token} - pros: {pros}, cons: {cons}. You decide what matters most."
    }
}

# Enhanced reply templates with formatting variations
REPLY_TEMPLATES = {
    'question': [
        "Is {token} looking bullish or bearish right now?",
        "Anyone else watching this {token} setup closely?",
        "How are we feeling about {token} after that announcement?",
        "Have you seen what's happening with {token}?",
        "Are we ignoring the obvious signals on {token} chart?",
        "When will {token} finally break this range?",
        "How does this affect the {token} thesis?"
    ],
    'observation': [
        "Interesting price action on {token} today.",
        "The {token} chart is telling a clear story right now.",
        "Something's brewing with {token} - volume profile changing.",
        "Market seems to be misreading the {token} developments.",
        "Sentiment shift happening in real-time for {token}.",
        "The {token} correlation with {other_token} breaking down.",
        "Smart money movement in {token} becoming obvious."
    ],
    'opinion': [
        "Unpopular opinion: {token} is {sentiment_adj} {sentiment}.",
        "Hot take: {token} will {prediction} before year end.",
        "My view: {token} fundamentals are {comparative} than price suggests.",
        "Been saying this for months - {token} is setting up for {outcome}.",
        "Against consensus here, but {token} looks {sentiment} to me.",
        "{token} thesis remains {status} despite recent {events}.",
        "Long-term, {token} still positioned for {future_outcome}."
    ],
    'analysis': [
        "Looking at {token} metrics: {metric1} up {percent1}%, {metric2} down {percent2}%.",
        "Three key factors for {token} right now: {factor1}, {factor2}, and {factor3}.",
        "The {token} chart shows clear {pattern} on the {timeframe} timeframe.",
        "On-chain analysis reveals {token} {on_chain_insight} pattern forming.",
        "Comparison between {token} and {other_token} reveals {comparison_insight}.",
        "Technical + fundamental confluence for {token}: {insight}.",
        "Market structure for {token} suggests {market_structure_insight}."
    ]
}

# Token pair comparison templates
TOKEN_COMPARISON_PHRASES = {
    'outperforming': [
        "{token1} absolutely crushing {token2} in performance lately",
        "Choosing {token1} over {token2} has been the winning trade",
        "The {token1}/{token2} ratio expanding as the former dominates",
        "{token1} strength vs {token2} weakness creating opportunities",
        "The divergence between {token1} and {token2} growing wider",
        "Rotation from {token2} into {token1} clearly visible",
        "{token1} making {token2} look stagnant by comparison",
        "The narrative shift from {token2} to {token1} playing out in price"
    ],
    'correlated': [
        "{token1} and {token2} moving in perfect lockstep",
        "High correlation between {token1} and {token2} continues",
        "The {token1}/{token2} pair maintaining stable ratio",
        "Trading the spread between {token1} and {token2} remains challenging",
        "Nearly identical price action between {token1} and {token2}",
        "Macro factors affecting {token1} and {token2} similarly",
        "The {token1}/{token2} chart showing strongest correlation in months",
        "Algorithmic trading linking {token1} and {token2} price movements"
    ],
    'diverging': [
        "{token1} and {token2} correlation breaking down",
        "Interesting divergence developing between {token1} and {token2}",
        "The {token1}/{token2} ratio at critical inflection point",
        "Sector rotation causing {token1} and {token2} paths to separate",
        "Fundamentals driving {token1} and {token2} in opposite directions",
        "The decoupling between {token1} and {token2} accelerating",
        "Arbitrage opportunity emerging in the {token1}/{token2} spread",
        "Technical setups for {token1} and {token2} showing opposite patterns"
    ],
    'flippening': [
        "The {token1}/{token2} flippening getting closer by the day",
        "Market cap ratio between {token1} and {token2} approaching parity",
        "{token1} challenging {token2}'s dominance in the {sector} sector",
        "Potential flippening between {token1} and {token2} gathering momentum",
        "The battle between {token1} and {token2} for {sector} dominance intensifies",
        "Historical market cap gap between {token1} and {token2} narrowing",
        "Narrative shifting to favor {token1} over incumbent {token2}",
        "The {token1} flippening {token2} narrative gaining credibility"
    ]
}

    # Helper functions for meme phrase generation

# Add these to meme_phrases.py

# Tech domains for integration with crypto content
class TechDomain(Enum):
    AI = "ai"
    QUANTUM = "quantum"
    BLOCKCHAIN = "blockchain"
    ML = "machine_learning"
    CLOUD = "cloud_computing"
    EDGE = "edge_computing"
    IOT = "internet_of_things"
    CYBERSECURITY = "cybersecurity"
    AR_VR = "ar_vr"
    ROBOTICS = "robotics"

# Tech trend sentiment categories
class TechTrend(Enum):
    EMERGING = "emerging"
    MAINSTREAM = "mainstream"
    DISRUPTIVE = "disruptive"
    SPECULATIVE = "speculative"
    MATURE = "mature"
    DECLINING = "declining"

# AI specific meme phrases
AI_PHRASES = {
    'bullish': [
        "AI capabilities growing faster than {token} prices!",
        "LLMs making gains that would make {token} traders jealous!",
        "AI adoption curve looking more vertical than {token}'s pump!",
        "Multimodal AI transforming tech faster than {token} transforms finance!",
        "AI processing capabilities 10x'ing annually - better than {token} returns!",
        "AGI speculation more bullish than even the most hardcore {token} maximalist!",
        "AI progress making {token} growth look like slow motion!",
        "AI market cap growth giving even {token} a run for its money!",
        "Smart contracts meet smarter AI - {token}'s perfect companion!",
        "AI tools building better trading algorithms for {token} than humans!"
    ],
    'bearish': [
        "AI winter fears spreading FUD worse than {token} crashes",
        "Regulation hitting AI harder than it hit {token} markets",
        "AI complexity becoming a barrier just like {token} UX problems",
        "AI overpromising and underdelivering, reminds me of some {token} projects",
        "AI hype cycle reaching peak disillusionment, like {token} after 2017",
        "AI resource costs rising faster than {token} gas fees",
        "AI alignment problems are the AI equivalent of {token}'s scalability issues",
        "Centralized AI control as concerning as centralized {token} exchanges",
        "AI hallucinations as unpredictable as {token} price movements",
        "OpenAI drama more chaotic than {token} governance proposals"
    ],
    'neutral': [
        "AI models and {token} networks - both scaling gradually",
        "AI research continuing steadily, much like {token} development",
        "The AI market stabilizing similar to {token}'s maturation process",
        "AI capabilities improving incrementally, just like {token} protocols",
        "AI adoption happening at a measured pace, similar to {token} institutional adoption",
        "AI infrastructure growing organically like {token}'s node network",
        "AI standards developing similarly to {token} protocols",
        "AI and {token} both finding their appropriate value propositions",
        "AI progress continuing despite the noise, like {token} development",
        "AI models and {token} networks both require patience to mature"
    ],
    'volatile': [
        "AI capabilities spiking and crashing like {token} in a volatile market!",
        "AI breakthroughs and setbacks cycling faster than {token} price swings!",
        "AI ethics debates as heated and unpredictable as {token} market moves!",
        "AI progress showing wild oscillations that make {token} charts look stable!",
        "AI research results ping-ponging more erratically than {token} during liquidation cascades!",
        "AI benchmark leaderboards flipping positions faster than {token} market rankings!",
        "AI sentiment swinging from euphoria to panic faster than {token} traders!",
        "AI field as turbulent and chaotic as {token} during a flash crash!",
        "AI capabilities jumping and dropping like {token} on leverage!",
        "AI model performance as inconsistent as {token} price predictions!"
    ],
    'recovering': [
        "AI interest rebounding after setbacks, like {token} finding a bottom",
        "AI budgets starting to recover, similar to {token} after bear markets",
        "AI startups getting funded again, like {token} projects after winter",
        "AI research momentum picking up, resembling {token}'s accumulation phase",
        "AI showing green shoots after disappointments, like {token} post-correction",
        "AI metrics improving gradually, similar to {token}'s organic recovery",
        "AI conference attendance growing again, like {token} community engagement after crashes",
        "AI model performance bouncing back from plateaus, like {token} recovering support",
        "AI adoption metrics starting to climb, mirroring {token}'s recovery phase",
        "AI venture funding showing signs of life, like {token} markets after capitulation"
    ]
}

# Quantum computing phrases
QUANTUM_PHRASES = {
    'bullish': [
        "Quantum computers threatening to break {token}'s cryptography!",
        "Quantum supremacy achieved faster than {token} achieved mainstream recognition!",
        "Quantum qubits scaling up more impressively than {token}'s TPS!",
        "Quantum computing and {token} - the two exponential technologies of our era!",
        "Quantum algorithms solving problems faster than {token} confirms transactions!",
        "Quantum entanglement more mysterious than {token}'s price movements!",
        "Quantum computing progressing faster than {token} layer 2 solutions!",
        "Quantum resistance becoming as important as holding {token}!",
        "Quantum computing investments rivaling early {token} investments in potential returns!",
        "Quantum hardware improvements outpacing {token} protocol upgrades!"
    ],
    'bearish': [
        "Quantum computing challenges making {token}'s scaling issues look trivial",
        "Quantum error rates still too high, like {token}'s failed transactions",
        "Quantum hardware limitations as restrictive as {token}'s throughput constraints",
        "Quantum funding winter could be worse than {token}'s bear market",
        "Quantum timeline setbacks more disappointing than delayed {token} upgrades",
        "Quantum coherence problems as persistent as {token}'s security vulnerabilities",
        "Quantum computing's practical applications as elusive as {token}'s mainstream use cases",
        "Quantum computing skepticism growing similar to {token} criticism",
        "Quantum industry consolidation leaving fewer players, like {token} exchange monopolies",
        "Quantum research plateaus as concerning as {token} development slowdowns"
    ],
    'neutral': [
        "Quantum computing progress measured in qubits, while {token} progress measured in adoption",
        "Quantum research continuing alongside {token} development - both playing the long game",
        "Quantum computing and {token} both needing patience from investors",
        "Quantum milestones achieved steadily, similar to {token}'s roadmap progress",
        "Quantum applications expanding gradually like {token} use cases",
        "Quantum error correction improving incrementally, like {token} transaction efficiency",
        "Quantum computing talent growing steadily like {token} developer communities",
        "Quantum algorithms and {token} protocols both maturing at their own pace",
        "Quantum hardware and {token} infrastructure both facing similar scaling challenges",
        "Quantum computing standards developing similar to {token} protocol standards"
    ],
    'volatile': [
        "Quantum breakthrough announcements as chaotic as {token} flash pumps!",
        "Quantum computing claims fluctuating more wildly than {token} price predictions!",
        "Quantum research results as unpredictable as {token} during high volatility!",
        "Quantum news cycle creating whipsaws that rival {token} market movements!",
        "Quantum computing progress spiking and crashing like {token} order books!",
        "Quantum qubit quality metrics bouncing around like {token} during a trading range!",
        "Quantum industry sentiment swinging like {token} during FOMC announcements!",
        "Quantum hardware performance metrics as oscillating as {token} intraday charts!",
        "Quantum startups' valuations popping and dropping like {token} microcaps!",
        "Quantum computing roadmaps changing as frequently as {token} technical indicators!"
    ],
    'recovering': [
        "Quantum computing regaining momentum after setbacks, like {token} in recovery",
        "Quantum research funding starting to flow again, similar to {token} after bear market",
        "Quantum error rates improving after plateaus, like {token} breaking resistance",
        "Quantum hardware reliability bouncing back, reminiscent of {token} reclaiming support",
        "Quantum milestone achievements resuming, similar to {token}'s price recovery",
        "Quantum computing enthusiasm returning gradually, like {token} after capitulation",
        "Quantum tech talent returning to the field, like traders returning to {token}",
        "Quantum computing media coverage turning positive, like {token} sentiment shifts",
        "Quantum startups raising funds again, similar to {token} projects during recovery",
        "Quantum computing conferences growing again, like {token} community during early bull market"
    ]
}

# Blockchain technology phrases beyond trading
BLOCKCHAIN_TECH_PHRASES = {
    'bullish': [
        "Zero-knowledge proofs advancing faster than {token} price during bull runs!",
        "Layer 2 throughput reaching levels that make {token} maxis jealous!",
        "Blockchain interoperability making more progress than {token} has made in years!",
        "Decentralized identities could become more valuable than your {token} holdings!",
        "Smart contract innovations making {token}'s technology look primitive!",
        "Blockchain scaling solutions would pump harder than {token} if they had tokens!",
        "Decentralized storage growing faster than {token}'s market cap!",
        "Cross-chain bridges handling more value than {token}'s entire market cap!",
        "Blockchain oracle reliability becoming more critical than {token} price feeds!",
        "Consensus algorithm innovations that could make {token} obsolete if not adopted!"
    ],
    'bearish': [
        "Blockchain trilemma still unsolved, haunting {token} like all others",
        "On-chain privacy facing more restrictions than {token} exchange regulations",
        "Blockchain interoperability challenges more complex than {token}'s scaling issues",
        "Blockchain governance failures as damaging as {token} protocol vulnerabilities",
        "Cross-chain bridge hacks draining more value than {token} market dumps",
        "Zero-knowledge proof limitations as restrictive as {token} transaction throughput",
        "Layer 2 adoption slower than even {token}'s most pessimistic projections",
        "Blockchain developer exodus more concerning than {token} whale movements",
        "Blockchain energy consumption criticisms as persistent as {token} ponzi accusations",
        "Blockchain standardization as fragmented as the {token} ecosystem"
    ],
    'neutral': [
        "Layer 2 solutions growing steadily, much like {token}'s ecosystem",
        "Zero-knowledge applications expanding gradually alongside {token} use cases",
        "Blockchain interoperability advancing incrementally, similar to {token} development",
        "Decentralized identity implementations maturing at the same pace as {token} adoption",
        "Consensus mechanism research continuing parallel to {token} protocol upgrades",
        "Blockchain governance experiments evolving like {token} community structures",
        "Cross-chain messaging protocols developing alongside {token} standards",
        "Blockchain storage solutions scaling gradually like {token} networks",
        "Blockchain privacy tools improving incrementally like {token} security features",
        "Blockchain analytics capabilities expanding similar to {token} market intelligence"
    ],
    'volatile': [
        "Layer 2 TVL swinging wildly like {token} during leverage liquidations!",
        "Cross-chain bridge statistics as unstable as {token} during flash crashes!",
        "Blockchain governance votes as unpredictable as {token} whale movements!",
        "Zero-knowledge implementation metrics fluctuating more than {token} during FUD!",
        "Blockchain interoperability project rankings changing faster than {token} market cap ranks!",
        "Consensus algorithm benchmark results as inconsistent as {token} technical patterns!",
        "Blockchain bandwidth metrics spiking and crashing like {token} intraday!",
        "Blockchain developer activity as oscillating as {token} sentiment indicators!",
        "Blockchain conference attendance numbers as volatile as {token} trading volume!",
        "Blockchain research results as contradictory as {token} price predictions!"
    ],
    'recovering': [
        "Blockchain development activity recovering after quieter periods, like {token} after corrections",
        "Layer 2 adoption metrics showing renewed growth, similar to {token}'s price recovery",
        "Zero-knowledge implementation interest returning, like {token} buyer confidence",
        "Blockchain interoperability projects gaining momentum again, reminiscent of {token} accumulation phases",
        "Cross-chain transaction volumes picking up, similar to {token} trading activity recovery",
        "Blockchain governance participation increasing, like {token} community engagement after bear market",
        "Blockchain research funding starting to flow again, similar to {token} liquidity returning",
        "Consensus innovation papers being published again, like {token} positive development news returning",
        "Blockchain job openings trending upward, like {token} exchange listings after winter",
        "Blockchain startup funding showing early signs of recovery, like {token} forming a bottom"
    ]
}

# Machine learning phrases
ML_PHRASES = {
    'bullish': [
        "Machine learning models scaling faster than {token}'s network growth!",
        "ML inference speeds dropping faster than {token} transaction times!",
        "Neural networks growing deeper than {token} order books!",
        "ML model parameters increasing faster than {token} wallet addresses!",
        "Reinforcement learning beating markets better than any {token} trader!",
        "ML benchmarks rising more consistently than {token} yearly lows!",
        "ML infrastructure costs dropping faster than {token} transaction fees!",
        "Transfer learning more efficient than cross-chain {token} bridges!",
        "ML model compression techniques more impressive than {token} scalability solutions!",
        "ML research papers being published faster than {token} price predictions!"
    ],
    'bearish': [
        "ML compute requirements growing more unaffordable than {token} validator hardware",
        "ML model training costs exceeding {token} mining profitability",
        "ML reproducibility crisis as concerning as {token} protocol security issues",
        "ML talent concentration more centralized than {token} mining pools",
        "ML model bias problems as persistent as {token} volatility issues",
        "ML energy consumption drawing criticism similar to {token} environmental concerns",
        "ML interpretability challenges as difficult as explaining {token} value to no-coiners",
        "ML dataset quality issues as problematic as {token} market manipulation",
        "ML research plateaus as disappointing as delayed {token} roadmap items",
        "ML hardware limitations as restrictive as {token} blockchain throughput"
    ],
    'neutral': [
        "ML model architectures evolving steadily like {token} protocol improvements",
        "ML benchmark progress continuing at a measured pace, similar to {token} adoption metrics",
        "ML applications expanding gradually like {token} use cases",
        "ML research continuing alongside {token} development",
        "ML frameworks maturing similar to {token} development tools",
        "ML talent growing organically like {token} developer communities",
        "ML hardware advancing incrementally like {token} node requirements",
        "ML deployment methods standardizing like {token} interfaces",
        "ML ethics considerations developing alongside {token} governance models",
        "ML enterprise adoption progressing steadily like {token} institutional interest"
    ],
    'volatile': [
        "ML benchmark leaderboards changing faster than {token} market cap rankings!",
        "ML research directions shifting more abruptly than {token} investor sentiment!",
        "ML model performance metrics swinging wildly like {token} during market uncertainty!",
        "ML conference acceptance rates fluctuating more than {token} support levels!",
        "ML startup valuations as unpredictable as {token} during high volatility periods!",
        "ML hiring trends more erratic than {token} volume profiles!",
        "ML framework popularity charts looking like {token} with 100x leverage!",
        "ML research funding as inconsistent as {token} market liquidity!",
        "ML hardware benchmark results as confusing as {token} technical patterns!",
        "ML researcher attention shifting faster than {token} trader focus!"
    ],
    'recovering': [
        "ML research funding starting to recover like {token} after market bottoms",
        "ML startup activity showing signs of life similar to {token} after corrections",
        "ML model performance improving again like {token} reclaiming support levels",
        "ML conference submissions increasing similar to {token} trading volume during recovery",
        "ML job market warming up again, reminiscent of {token} renewed interest phases",
        "ML hardware sales trending upward like {token} breaking resistance",
        "ML research progress accelerating again, similar to {token} in early bull markets",
        "ML benchmark achievements resuming after plateaus, like {token} price action after accumulation",
        "ML industry outlook brightening like {token} sentiment shifts",
        "ML implementation case studies growing again, similar to {token} use case expansion after bear markets"
    ]
}

# AR/VR/XR phrases
AR_VR_PHRASES = {
    'bullish': [
        "Spatial computing adoption growing faster than {token} addresses!",
        "XR hardware sales outpacing {token} wallet growth!",
        "VR resolution improvements more impressive than {token} TPS gains!",
        "AR market projections more vertical than {token}'s bullish patterns!",
        "Metaverse land sales rivaling early {token} ROI!",
        "XR developer growth outpacing {token} developer communities!",
        "VR haptic feedback advancing faster than {token} UX improvements!",
        "AR glasses becoming as essential as having {token} in your portfolio!",
        "Spatial audio quality leaping ahead faster than {token} scaling solutions!",
        "XR enterprise adoption showing clearer ROI than most {token} use cases!"
    ],
    'bearish': [
        "VR adoption slower than even the most pessimistic {token} projections",
        "AR hardware limitations as restrictive as {token} throughput constraints",
        "XR content quality issues as disappointing as overhyped {token} projects",
        "Metaverse engagement metrics as concerning as declining {token} active addresses",
        "VR motion sickness problems as persistent as {token} volatility issues",
        "AR privacy concerns as serious as {token} regulatory threats",
        "XR developer exodus more rapid than {token} bear market capitulation",
        "Spatial computing standards as fragmented as {token} protocol compatibility",
        "VR consumer interest declining faster than {token} during extended corrections",
        "XR hardware costs as prohibitive as {token} validator requirements"
    ],
    'neutral': [
        "VR hardware improving steadily like {token} protocol upgrades",
        "AR adoption growing at a measured pace, similar to {token} institutional interest",
        "XR developer ecosystems maturing alongside {token} communities",
        "Metaverse standards developing similar to {token} interoperability protocols",
        "VR content libraries expanding gradually like {token} use cases",
        "AR enterprise implementations progressing like {token} business adoption",
        "XR user experience improving incrementally like {token} interfaces",
        "Spatial computing infrastructure building steadily like {token} node networks",
        "VR resolution and refresh rates climbing like {token}'s gradual adoption metrics",
        "AR glasses form factors evolving similar to {token} wallet usability"
    ],
    'volatile': [
        "VR hardware sales numbers bouncing around like {token} during market uncertainty!",
        "AR adoption metrics swinging wildly like {token} during FUD cycles!",
        "XR startup valuations as unpredictable as {token} during leverage liquidations!",
        "Metaverse token prices more volatile than even {token} during flash events!",
        "VR usage statistics fluctuating more than {token} exchange volumes!",
        "AR developer sentiment changing directions faster than {token} technical indicators!",
        "XR enterprise pilot project outcomes as inconsistent as {token} price predictions!",
        "VR content platform rankings shifting faster than {token} market cap positions!",
        "AR glasses pre-order numbers as erratic as {token} during rumor-driven trading!",
        "XR industry conference messaging as contradictory as {token} investor sentiment!"
    ],
    'recovering': [
        "VR headset sales showing signs of life, like {token} volume returning after capitulation",
        "AR developer activity picking up again, similar to {token} address growth during recovery",
        "XR investment starting to flow after winter, like {token} markets after bottoming",
        "Metaverse engagement metrics ticking up, reminiscent of {token} transaction count recoveries",
        "VR content creation accelerating again, like {token} development during accumulation phases",
        "AR enterprise implementations resuming, similar to {token} institutional interest returning",
        "XR hardware roadmaps becoming active again, like {token} protocol development after quiet periods",
        "Spatial computing startups raising funds again, similar to {token} projects during early bull markets",
        "VR platform user growth turning positive, like {token} breaking out of downtrends",
        "AR glasses pre-orders increasing, similar to {token} exchange net inflows during recovery"
    ]
}

# Integration with tech phrases
TECH_INTEGRATION_PHRASES = {
    'synergy': [
        "{token} and AI combining for next-level financial intelligence",
        "Quantum-resistant {token} cryptography becoming a competitive edge",
        "{token} DeFi protocols leveraging ML for better risk assessment",
        "AR visualization tools making {token} charts more intuitive than ever",
        "{token} wallets with built-in AI security detection",
        "Quantum random number generation improving {token} security primitives",
        "{token} networks utilizing ML for optimized transaction routing",
        "XR interfaces revolutionizing how we interact with {token} protocols",
        "AI-powered {token} trading becoming the new meta",
        "Blockchain and quantum cryptography creating ultra-secure {token} solutions"
    ],
    'disruption': [
        "AI might make current {token} oracles obsolete",
        "Quantum computers could break {token}'s cryptography before it adapts",
        "ML-based market manipulation harder to detect than typical {token} schemes",
        "XR universes creating digital economies that compete with {token} networks",
        "AI-generated content flooding {token} NFT marketplaces",
        "Quantum mining could render traditional {token} consensus mechanisms obsolete",
        "ML prediction markets more accurate than {token} futures",
        "AR overlays revealing hidden {token} market information asymmetries",
        "AI autonomous agents becoming better {token} traders than humans",
        "Quantum financial modeling making current {token} technical analysis look primitive"
    ],
    'evolution': [
        "{token} gradually incorporating AI for protocol governance decisions",
        "Quantum-resistant algorithms being carefully added to {token} roadmaps",
        "{token} trading interfaces slowly adopting ML-based tools",
        "XR experiences starting to feature {token} in more immersive ways",
        "{token} analytics dashboards evolving with AI-powered insights",
        "Quantum security measures becoming a consideration in {token} development",
        "{token} DeFi risk models incrementally improving with ML techniques",
        "AR information layers being tested with {token} transaction visualization",
        "{token} educational content increasingly using AI to personalize learning",
        "Quantum-inspired algorithms beginning to influence {token} protocol design"
    ],
    'speculation': [
        "Imagine if {token} incorporated AGI for autonomous governance!",
        "Could quantum mining create perfect {token} distribution?",
        "What if ML could predict {token} prices with 90% accuracy?",
        "Future XR economies might make {token} look like ancient technology",
        "AI-managed {token} portfolios could outperform all human traders",
        "Quantum-based {token} networks might achieve million TPS throughput",
        "ML-optimized smart contracts could make current {token} DeFi look primitive",
        "AR glasses might someday display everyone's {token} holdings above their heads",
        "AI could create and run DAOs better than human {token} communities",
        "Quantum teleportation principles might someday eliminate {token} bridge risks"
    ]
}

# Function to generate tech-focused phrases
def get_tech_phrase(token: str, tech_domain: TechDomain, sentiment: str = 'neutral', tech_trend: TechTrend = None) -> str:
    """
    Generate a tech-focused phrase related to a specific domain and sentiment
    
    Args:
        token (str): Cryptocurrency token to reference
        tech_domain (TechDomain): Technology domain for the phrase
        sentiment (str): Sentiment/mood for phrase tone
        tech_trend (TechTrend, optional): Technology trend category
        
    Returns:
        str: Generated tech phrase
    """
    # Select appropriate phrase collection based on domain
    if tech_domain == TechDomain.AI:
        phrases = AI_PHRASES
    elif tech_domain == TechDomain.QUANTUM:
        phrases = QUANTUM_PHRASES
    elif tech_domain == TechDomain.BLOCKCHAIN:
        phrases = BLOCKCHAIN_TECH_PHRASES
    elif tech_domain == TechDomain.ML:
        phrases = ML_PHRASES
    elif tech_domain == TechDomain.AR_VR:
        phrases = AR_VR_PHRASES
    else:
        # Default to AI phrases for other domains
        phrases = AI_PHRASES
    
    # Select appropriate sentiment category or default to neutral
    if sentiment in phrases:
        selected_phrases = phrases[sentiment]
    else:
        selected_phrases = phrases['neutral']
    
    # Select a random phrase and format with token
    selected_phrase = random.choice(selected_phrases)
    
    # If trend specified and using integration phrases
    if tech_trend and tech_domain == TechDomain.BLOCKCHAIN:
        trend_value = tech_trend.value if hasattr(tech_trend, 'value') else tech_trend
        if trend_value in TECH_INTEGRATION_PHRASES:
            integration_phrases = TECH_INTEGRATION_PHRASES[trend_value]
            integration_phrase = random.choice(integration_phrases)
            return integration_phrase.format(token=token)
    
    return selected_phrase.format(token=token)

# Enhanced MemePhraseGenerator method
def generate_tech_integrated_phrase(token: str, mood: Any, tech_domain: TechDomain = None, 
                                   tech_trend: TechTrend = None, additional_context: Dict[str, Any] = None) -> str:
    """
    Generate a phrase integrating tech and crypto themes
    
    Args:
        token: Token/chain symbol (e.g., 'BTC', 'ETH')
        mood: Mood object or string representing mood
        tech_domain: Optional technology domain to focus on
        tech_trend: Optional technology trend category
        additional_context: Additional context for more specific phrases
            
    Returns:
        Generated phrase integrating tech and crypto
    """
    # Extract mood value from object if needed
    mood_str = mood.value if hasattr(mood, 'value') else str(mood)
    
    # If tech domain specified, use tech-specific phrases
    if tech_domain:
        domain_value = tech_domain.value if hasattr(tech_domain, 'value') else tech_domain
        return get_tech_phrase(token, domain_value, mood_str, tech_trend)
    
    # If no tech domain specified but we want tech content, randomly select one
    elif additional_context and additional_context.get('include_tech', False):
        # Random selection from available domains
        domains = list(TechDomain)
        random_domain = random.choice(domains)
        return get_tech_phrase(token, random_domain, mood_str, tech_trend)
    
    # Otherwise fall back to standard meme phrase generation
    return MemePhraseGenerator.generate_meme_phrase(token, mood, additional_context)

def get_token_meme_phrase(token: str, context_type: str, context_value: str) -> str:
    """Generate meme phrases based on token, context type and value"""
    if context_type == 'mood':
        if context_value in TOKEN_MEME_PHRASES:
            phrases = TOKEN_MEME_PHRASES[context_value]
        else:
            phrases = TOKEN_MEME_PHRASES['neutral']
    elif context_type == 'volume':
        if context_value in VOLUME_PHRASES:
            phrases = VOLUME_PHRASES[context_value]
        else:
            phrases = VOLUME_PHRASES['stable']
    elif context_type == 'market_comparison':
        if context_value in MARKET_COMPARISON_PHRASES:
            phrases = MARKET_COMPARISON_PHRASES[context_value]
        else:
            phrases = MARKET_COMPARISON_PHRASES['correlated']
    elif context_type == 'smart_money':
        if context_value in SMART_MONEY_PHRASES:
            phrases = SMART_MONEY_PHRASES[context_value]
        else:
            phrases = SMART_MONEY_PHRASES['accumulating']
    elif context_type == 'technical':
        if context_value in TECHNICAL_ANALYSIS_PHRASES:
            phrases = TECHNICAL_ANALYSIS_PHRASES[context_value]
        else:
            phrases = TECHNICAL_ANALYSIS_PHRASES['indicators']
    elif context_type == 'defi':
        if context_value in DEFI_PHRASES:
            phrases = DEFI_PHRASES[context_value]
        else:
            phrases = DEFI_PHRASES['yield_farming']
    elif context_type == 'nft':
        if context_value in NFT_PHRASES:
            phrases = NFT_PHRASES[context_value]
        else:
            phrases = NFT_PHRASES['collections']
    elif context_type == 'layer1':
        if context_value in LAYER1_PHRASES:
            phrases = LAYER1_PHRASES[context_value]
        else:
            phrases = LAYER1_PHRASES['ecosystem']
    elif context_type == 'regulatory':
        if context_value in REGULATORY_NEWS_PHRASES:
            phrases = REGULATORY_NEWS_PHRASES[context_value]
        else:
            phrases = REGULATORY_NEWS_PHRASES['regulation']
    else:
        # Default to neutral mood if specific context not found
        phrases = TOKEN_MEME_PHRASES['neutral']
    
    # Select a random phrase and format it with the token
    selected_phrase = random.choice(phrases)
    
    # Format with token if template contains {token}
    if '{token}' in selected_phrase:
        return selected_phrase.format(token=token)
    else:
        return selected_phrase

def get_token_comparison_phrase(token1: str, token2: str, comparison_type: str) -> str:
    """Generate phrases comparing two tokens"""
    if comparison_type in TOKEN_COMPARISON_PHRASES:
        phrases = TOKEN_COMPARISON_PHRASES[comparison_type]
    else:
        # Fallback to diverging if specific comparison not found
        phrases = TOKEN_COMPARISON_PHRASES['diverging']
    
    selected_phrase = random.choice(phrases)
    return selected_phrase.format(token1=token1, token2=token2, sector="crypto")

def get_audience_targeted_phrase(token: str, audience_type: str, sentiment: str) -> str:
    """Generate phrases targeted to specific audience types"""
    # Default values for template placeholders
    context = {
        'token': token,
        'level': f"${random.randint(10, 100)}K",
        'support': f"${random.randint(5, 50)}K",
        'target': f"${random.randint(15, 150)}K",
        'range_low': f"${random.randint(5, 50)}K",
        'range_high': f"${random.randint(55, 150)}K",
        'pros': "strong tech, growing adoption",
        'cons': "regulatory uncertainty, competition"
    }
    
    # Get appropriate audience template
    if audience_type in AUDIENCE_TEMPLATES:
        if sentiment in AUDIENCE_TEMPLATES[audience_type]:
            template = AUDIENCE_TEMPLATES[audience_type][sentiment]
        else:
            template = AUDIENCE_TEMPLATES[audience_type]['neutral']
    else:
        # Fallback to investor neutral
        template = AUDIENCE_TEMPLATES['investor']['neutral']
    
    # Format the template with context
    try:
        return template.format(**context)
    except KeyError:
        # Fallback if template has missing keys
        return f"{token} remains an interesting project to watch in the current market conditions."

def get_timeframe_phrase(token: str, timeframe: str) -> str:
    """Generate phrases related to specific timeframes"""
    if timeframe in TIME_CONTEXT_PHRASES:
        phrases = TIME_CONTEXT_PHRASES[timeframe]
    else:
        # Fallback to mid-term if specific timeframe not found
        phrases = TIME_CONTEXT_PHRASES['mid_term']
    
    selected_phrase = random.choice(phrases)
    
    # Format with token if template contains {token}
    if '{token}' in selected_phrase:
        return selected_phrase.format(token=token)
    else:
        return selected_phrase

def get_meme_culture_phrase(token: str, meme_type: str) -> str:
    """Generate phrases based on meme culture"""
    if meme_type in MEME_CULTURE_PHRASES:
        phrases = MEME_CULTURE_PHRASES[meme_type]
    else:
        # Fallback to classic memes if specific type not found
        phrases = MEME_CULTURE_PHRASES['classic_memes']
    
    selected_phrase = random.choice(phrases)
    
    # Format with token if template contains {token}
    if '{token}' in selected_phrase:
        return selected_phrase.format(token=token)
    else:
        return selected_phrase

def get_market_cycle_phrase(token: str, cycle_phase: str) -> str:
    """Generate phrases based on market cycle phase"""
    if cycle_phase in MARKET_CYCLE_PHRASES:
        phrases = MARKET_CYCLE_PHRASES[cycle_phase]
    else:
        # Fallback to accumulation if specific phase not found
        phrases = MARKET_CYCLE_PHRASES['accumulation']
    
    selected_phrase = random.choice(phrases)
    
    # Format with token if template contains {token}
    if '{token}' in selected_phrase:
        return selected_phrase.format(token=token)
    else:
        return selected_phrase

def get_market_psychology_phrase(token: str, psychology_state: str) -> str:
    """Generate phrases based on market psychology"""
    if psychology_state in MARKET_PSYCHOLOGY_PHRASES:
        phrases = MARKET_PSYCHOLOGY_PHRASES[psychology_state]
    else:
        # Fallback to fear if specific state not found
        phrases = MARKET_PSYCHOLOGY_PHRASES['fear']
    
    selected_phrase = random.choice(phrases)
    
    # Format with token if template contains {token}
    if '{token}' in selected_phrase:
        return selected_phrase.format(token=token)
    else:
        return selected_phrase

# MemePhraseGenerator class - updated with more generation options
class MemePhraseGenerator:
    """
    Enhanced generator for meme phrases based on context
    Provides a consistent way to generate diverse meme phrases for any token
    """
    
    @staticmethod
    def generate_meme_phrase(token: str, mood: Any, additional_context: Dict[str, Any] = None) -> str:
        """
        Generate a meme phrase for a specific token and mood with expanded context options
        
        Args:
            token: Token/chain symbol (e.g., 'BTC', 'ETH')
            mood: Mood object or string representing mood
            additional_context: Additional context for more specific phrases
            
        Returns:
            Generated meme phrase
        """
        # Extract mood value from object if needed
        mood_str = mood.value if hasattr(mood, 'value') else str(mood)
        
        # Handle special context types
        if additional_context:
            # Volume context
            if 'volume_trend' in additional_context:
                return get_token_meme_phrase(token, 'volume', additional_context['volume_trend'])
                
            # Market comparison context
            if 'market_comparison' in additional_context:
                return get_token_meme_phrase(token, 'market_comparison', additional_context['market_comparison'])
                
            # Smart money context
            if 'smart_money' in additional_context:
                return get_token_meme_phrase(token, 'smart_money', additional_context['smart_money'])
                
            # Technical analysis context
            if 'technical' in additional_context:
                tech_type = additional_context.get('technical_type', 'indicators')
                return get_token_meme_phrase(token, 'technical', tech_type)
                
            # DeFi specific context
            if 'defi' in additional_context:
                defi_type = additional_context.get('defi_type', 'yield_farming')
                return get_token_meme_phrase(token, 'defi', defi_type)
                
            # NFT specific context
            if 'nft' in additional_context:
                nft_type = additional_context.get('nft_type', 'collections')
                return get_token_meme_phrase(token, 'nft', nft_type)
                
            # Layer 1 specific context
            if 'layer1' in additional_context:
                layer1_type = additional_context.get('layer1_type', 'ecosystem')
                return get_token_meme_phrase(token, 'layer1', layer1_type)
                
            # Token comparison context
            if 'comparison_token' in additional_context:
                comparison_type = additional_context.get('comparison_type', 'diverging')
                token2 = additional_context['comparison_token']
                return get_token_comparison_phrase(token, token2, comparison_type)
                
            # Audience targeting
            if 'audience' in additional_context:
                audience_type = additional_context['audience']
                sentiment = additional_context.get('sentiment', mood_str)
                return get_audience_targeted_phrase(token, audience_type, sentiment)
                
            # Time context targeting
            if 'timeframe' in additional_context:
                return get_timeframe_phrase(token, additional_context['timeframe'])
                
            # Meme culture targeting
            if 'meme_culture' in additional_context:
                meme_type = additional_context.get('meme_type', 'classic_memes')
                return get_meme_culture_phrase(token, meme_type)
                
            # Market cycle context
            if 'market_cycle' in additional_context:
                return get_market_cycle_phrase(token, additional_context['market_cycle'])
                
            # Market psychology context
            if 'market_psychology' in additional_context:
                return get_market_psychology_phrase(token, additional_context['market_psychology'])
                
            # Regulatory context
            if 'regulatory' in additional_context:
                regulatory_type = additional_context.get('regulatory_type', 'regulation')
                return get_token_meme_phrase(token, 'regulatory', regulatory_type)
        
        # Default to mood-based phrase
        return get_token_meme_phrase(token, 'mood', mood_str)

    @staticmethod
    def generate_formatted_reply(token: str, template_type: str, additional_context: Dict[str, Any] = None) -> str:
        """
        Generate a formatted reply using templates
        
        Args:
            token: Token/chain symbol
            template_type: Type of template to use
            additional_context: Additional context for template filling
            
        Returns:
            Formatted reply text
        """
        # Select template based on type
        if template_type in REPLY_TEMPLATES:
            templates = REPLY_TEMPLATES[template_type]
            template = random.choice(templates)
        else:
            # Fallback to observation
            templates = REPLY_TEMPLATES['observation']
            template = random.choice(templates)
            
        # Process context values
        context = {
            'token': token.upper()
        }
        
        # Add additional context
        if additional_context:
            context.update(additional_context)
            
        # Add random sentiment if needed
        if '{sentiment}' in template and 'sentiment' not in context:
            sentiments = ['bullish', 'bearish', 'neutral', 'uncertain']
            context['sentiment'] = random.choice(sentiments)
            
        # Add sentiment adjective if needed
        if '{sentiment_adj}' in template and 'sentiment_adj' not in context:
            mood = context.get('sentiment', 'neutral')
            amplifiers = SENTIMENT_AMPLIFIERS.get(mood, SENTIMENT_AMPLIFIERS['intensity'])
            context['sentiment_adj'] = random.choice(amplifiers)
            
        # Try to format template with available context
        try:
            return template.format(**context)
        except KeyError:
            # Missing key, return simple fallback
            return f"Interesting developments with {token} happening right now."
            
    @staticmethod
    def generate_diverse_phrase(token: str, sentiment: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Main function to generate diverse phrases with rich context options
    
        Args:
            token: Token symbol
            sentiment: Optional sentiment override
            context: Optional contextual parameters
        
        Returns:
            Generated diverse phrase
        """
        # Initialize context if not provided
        if context is None:
            context = {}
        
        # Select random sentiment if not specified
        if sentiment is None:
            sentiment_options = list(MarketSentiment)
            sentiment = random.choice(sentiment_options).value
        
        # Generate the phrase with all available context
        return MemePhraseGenerator.generate_meme_phrase(token, sentiment, context)
            
    @staticmethod
    def get_token_meme_phrase(token: str, context: str, subcontext: Optional[str] = None) -> str:
        """
        Get a random token-specific meme phrase based on context with expanded options
    
        Args:
            token: Token symbol or name
            context: Main context (mood, volume, market_comparison, etc.)
            subcontext: Sub-context for more specific phrases
        
        Returns:
            Random meme phrase for the given token and context
        """
        return get_token_meme_phrase(token, context, subcontext or "neutral")
        
    @staticmethod
    def get_token_comparison_phrase(token1: str, token2: str, comparison_type: str = "diverging") -> str:
        """
        Get a phrase comparing two tokens
    
        Args:
            token1: First token symbol
            token2: Second token symbol
            comparison_type: Type of comparison relationship
        
        Returns:
            Comparison phrase for the given tokens
        """
        return get_token_comparison_phrase(token1, token2, comparison_type)
    
    @staticmethod
    def get_audience_targeted_phrase(token: str, audience_type: str, sentiment: str = 'neutral') -> str:
        """
        Get a phrase targeted to a specific audience type
    
        Args:
            token: Token symbol
            audience_type: Type of audience to target
            sentiment: Sentiment tone to use
        
        Returns:
            Audience-targeted phrase
        """
        return get_audience_targeted_phrase(token, audience_type, sentiment)
    
    @staticmethod
    def get_timeframe_phrase(token: str, timeframe: str) -> str:
        """
        Get a timeframe-specific phrase for a token
    
        Args:
            token: Token symbol
            timeframe: Timeframe context
        
        Returns:
            Timeframe-appropriate phrase
        """
        return get_timeframe_phrase(token, timeframe)
    
    @staticmethod
    def get_market_cycle_phrase(token: str, cycle_phase: str) -> str:
        """
        Get a market cycle phase specific phrase
    
        Args:
            token: Token symbol
            cycle_phase: Market cycle phase
        
        Returns:
            Market cycle appropriate phrase
        """
        return get_market_cycle_phrase(token, cycle_phase)
    
    @staticmethod
    def get_market_psychology_phrase(token: str, psychology: str) -> str:
        """
        Get a market psychology specific phrase
    
        Args:
            token: Token symbol
            psychology: Market psychology phase
        
        Returns:
            Psychology-appropriate phrase
        """
        return get_market_psychology_phrase(token, psychology)
    
    @staticmethod
    def get_meme_culture_phrase(token: str, meme_type: str) -> str:
        """
        Get a meme culture specific phrase
    
        Args:
            token: Token symbol
            meme_type: Type of meme culture reference
        
        Returns:
            Meme culture appropriate phrase
        """
        return get_meme_culture_phrase(token, meme_type)

# Example usage
if __name__ == "__main__":
    # Create the generator
    generator = MemePhraseGenerator()
    
    # Example: Basic mood-based generation
    print("Basic mood-based generation:")
    print(generator.generate_meme_phrase("BTC", "bullish"))
    print(generator.generate_meme_phrase("ETH", MarketSentiment.BEARISH))
    print()
    
    # Example: Context-specific generation
    print("Context-specific generation:")
    print(generator.generate_meme_phrase("SOL", "neutral", {"technical": True, "technical_type": "patterns"}))
    print(generator.generate_meme_phrase("DOGE", "bullish", {"meme_culture": True, "meme_type": "classic_memes"}))
    print()
    
    # Example: Token comparison
    print("Token comparison:")
    print(generator.generate_meme_phrase("BTC", "neutral", {"comparison_token": "ETH", "comparison_type": "flippening"}))
    print()
    
    # Example: Formatted replies
    print("Formatted replies:")
    context = {
        "prediction": "reach new all-time highs",
        "timeframe": "Q4",
        "sentiment": "bullish",
        "other_token": "ETH"
    }
    print(generator.generate_formatted_reply("BTC", "question"))
    print(generator.generate_formatted_reply("ETH", "opinion", context))
    print(generator.generate_formatted_reply("LINK", "analysis", {
        "metric1": "active addresses", "percent1": 23,
        "metric2": "exchange outflows", "percent2": 15,
        "pattern": "bull flag", "timeframe": "daily"
    }))