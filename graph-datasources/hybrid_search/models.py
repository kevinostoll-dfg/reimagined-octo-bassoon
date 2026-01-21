"""
Pydantic models for structured weekly market analysis output.
Matches the WeeklyWrapUpData TypeScript interface.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class WeekRange(BaseModel):
    """Week date range"""
    start: str = Field(..., description="Start date in format 'Oct 14, 2024'")
    end: str = Field(..., description="End date in format 'Oct 21, 2024'")


class Performance(BaseModel):
    """Price performance metrics"""
    priceChange: float = Field(..., description="Price change in dollars")
    priceChangePercent: float = Field(..., description="Price change percentage")
    volume: str = Field(..., description="Volume as formatted string (e.g., '1.2B')")
    volumeChange: float = Field(..., description="Volume change percentage")
    high: float = Field(..., description="Weekly high price")
    low: float = Field(..., description="Weekly low price")


class KeyHighlight(BaseModel):
    """Key highlight item"""
    type: Literal["positive", "negative", "neutral"] = Field(..., description="Highlight type")
    category: str = Field(..., description="Category (e.g., 'Earnings', 'News', 'Analyst')")
    title: str = Field(..., description="Highlight title")
    description: str = Field(..., description="Highlight description")
    impact: Literal["high", "medium", "low"] = Field(..., description="Impact level")
    date: str = Field(..., description="Date in format 'Oct 15, 2024'")


class DailyPerformance(BaseModel):
    """Daily performance data"""
    day: str = Field(..., description="Day abbreviation (e.g., 'Mon', 'Tue')")
    date: str = Field(..., description="Date in format 'Oct 14, 2024'")
    price: float = Field(..., description="Closing price")
    change: float = Field(..., description="Price change in dollars")
    changePercent: float = Field(..., description="Price change percentage")
    volume: int = Field(..., description="Trading volume")


class SectorComparison(BaseModel):
    """Sector comparison metrics"""
    sectorPerformance: float = Field(..., description="Sector performance percentage")
    sectorAverage: float = Field(..., description="Sector average performance")
    rank: int = Field(..., description="Rank within sector (1 = best)")
    totalInSector: int = Field(..., description="Total companies in sector")


class TopStory(BaseModel):
    """Top news story"""
    title: str = Field(..., description="Story title")
    source: str = Field(..., description="News source")
    sentiment: Literal["bullish", "bearish", "neutral"] = Field(..., description="Sentiment")
    impact: str = Field(..., description="Impact description")


class NewsSummary(BaseModel):
    """News summary statistics"""
    totalNews: int = Field(..., description="Total news articles")
    bullish: int = Field(..., description="Number of bullish articles")
    bearish: int = Field(..., description="Number of bearish articles")
    neutral: int = Field(..., description="Number of neutral articles")
    topStory: TopStory = Field(..., description="Top story")


class AnalystActivity(BaseModel):
    """Analyst activity metrics"""
    upgrades: int = Field(..., description="Number of upgrades")
    downgrades: int = Field(..., description="Number of downgrades")
    initiations: int = Field(..., description="Number of initiations")
    priceTargetChanges: int = Field(..., description="Number of price target changes")
    averagePriceTarget: float = Field(..., description="Average price target")
    priceTargetChange: float = Field(..., description="Price target change percentage")


class LargestTrade(BaseModel):
    """Largest options trade"""
    type: Literal["CALL", "PUT"] = Field(..., description="Option type - must be 'CALL' or 'PUT'")
    strike: float = Field(..., description="Strike price")
    volume: int = Field(..., description="Trade volume")
    premium: str = Field(..., description="Premium as formatted string")


class OptionsActivity(BaseModel):
    """Options activity metrics"""
    unusualVolume: int = Field(..., description="Unusual volume count")
    callPutRatio: float = Field(..., description="Call/Put ratio")
    largestTrade: Optional[LargestTrade] = Field(
        default=None, 
        description="Largest trade (use None if no options data available, or provide with type='CALL' or 'PUT')"
    )


class TopBuyerSeller(BaseModel):
    """Top buyer or seller"""
    name: str = Field(..., description="Institution name")
    amount: str = Field(..., description="Amount as formatted string")


class InstitutionalActivity(BaseModel):
    """Institutional activity metrics"""
    netFlow: float = Field(..., description="Net flow in dollars")
    netFlowPercent: float = Field(..., description="Net flow percentage")
    topBuyer: TopBuyerSeller = Field(..., description="Top buyer")
    topSeller: TopBuyerSeller = Field(..., description="Top seller")


class SWOT(BaseModel):
    """SWOT analysis"""
    strengths: List[str] = Field(..., description="List of strengths")
    weaknesses: List[str] = Field(..., description="List of weaknesses")
    opportunities: List[str] = Field(..., description="List of opportunities")
    threats: List[str] = Field(..., description="List of threats")


class WeeklyWrapUpData(BaseModel):
    """Complete weekly market analysis data structure"""
    weekRange: WeekRange = Field(..., description="Week date range")
    performance: Performance = Field(..., description="Price performance metrics")
    keyHighlights: List[KeyHighlight] = Field(..., description="Key highlights")
    dailyPerformance: List[DailyPerformance] = Field(..., description="Daily performance breakdown")
    sectorComparison: SectorComparison = Field(..., description="Sector comparison")
    newsSummary: NewsSummary = Field(..., description="News summary")
    analystActivity: AnalystActivity = Field(..., description="Analyst activity")
    optionsActivity: OptionsActivity = Field(..., description="Options activity")
    institutionalActivity: InstitutionalActivity = Field(..., description="Institutional activity")
    swot: SWOT = Field(..., description="SWOT analysis")

