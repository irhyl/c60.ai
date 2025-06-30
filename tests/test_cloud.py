# from interface.cloud_runner import CloudRunner, CloudProvider

import pytest

@pytest.mark.skip(reason="CloudProvider/CloudRunner removed from codebase")
def test_cloudrunner_local():
    runner = CloudRunner(provider=CloudProvider.LOCAL)
    status = runner.run_pipeline(pipeline_config={"pipeline_id": "test"}, dataset_path="dummy.csv")
    assert isinstance(status, (str, dict))
    if isinstance(status, dict):
        assert status["status"] == "completed"
        assert status["provider"] == "local"
    else:
        assert status == "completed"


@pytest.mark.skip(reason="CloudProvider/CloudRunner removed from codebase")
def test_cloudrunner_cloud():
    for provider in [CloudProvider.AWS, CloudProvider.GCP, CloudProvider.AZURE]:
        runner = CloudRunner(provider=provider, config={})
        status = runner.run_pipeline(pipeline_config={"pipeline_id": "test"}, dataset_path="dummy.csv")
        assert isinstance(status, dict)
        assert status["status"] == "completed"
        assert status["provider"] == provider.value
