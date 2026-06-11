import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.redis_cache import (
    get_card_features,
    update_card_features,
)

def _make_mock_redis_with_pipeline() -> tuple[MagicMock, AsyncMock]:
    mock_pipe = AsyncMock()
    mock_pipe.hset = MagicMock()
    mock_pipe.expire = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=[True, True])

    mock_redis = MagicMock()
    mock_redis.pipeline.return_value.__aenter__ = AsyncMock(return_value=mock_pipe)
    mock_redis.pipeline.return_value.__aexit__ = AsyncMock(return_value=False)
    mock_redis.delete = AsyncMock()

    return mock_redis, mock_pipe

@pytest.mark.asyncio
async def test_get_card_features_returns_dict() -> None:
    mock_redis = AsyncMock()
    fake_data = {"amounts": "10.0,20.0", "sum_2": "30.0"}
    mock_redis.hgetall = AsyncMock(return_value=fake_data)

    result = await get_card_features(mock_redis, "card_1")
    assert result == fake_data
    mock_redis.hgetall.assert_called_once_with("card:card_1:features")

@pytest.mark.asyncio
async def test_get_card_features_empty() -> None:
    mock_redis = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={})

    result = await get_card_features(mock_redis, "card_2")
    assert result is None

@pytest.mark.asyncio
async def test_update_card_features_populates_hash() -> None:
    mock_redis, mock_pipe = _make_mock_redis_with_pipeline()
    
    # 3 items: w=2 slice has 2, w=4 slice has 3, w=9 slice has 3
    history = [(10.0, 1000.0), (20.0, 2000.0), (30.0, 3000.0)]
    
    await update_card_features(mock_redis, "card_1", history)
    
    mock_redis.pipeline.assert_called_once_with(transaction=True)
    
    # Verify mapping passed to hset
    hset_call = mock_pipe.hset.call_args
    assert hset_call is not None
    mapping = hset_call[1]["mapping"]
    
    assert mapping["amounts"] == "10.0,20.0,30.0"
    assert mapping["timestamps"] == "1000.0,2000.0,3000.0"
    
    # window size w=2 (from w-1)
    assert mapping["count_2"] == "2"
    assert mapping["sum_2"] == "50.0"  # 20 + 30
    assert float(mapping["sum_sq_2"]) == 1300.0  # 400 + 900
    assert mapping["min_2"] == "20.0"
    assert mapping["max_2"] == "30.0"
    
    # window size w=4 (which gets clipped to 3 items)
    assert mapping["count_4"] == "3"
    assert mapping["sum_4"] == "60.0"  # 10 + 20 + 30
    
    # check delete if history is empty
    await update_card_features(mock_redis, "card_1", [])
    mock_redis.delete.assert_called_once_with("card:card_1:features")
