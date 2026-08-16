import pytest


@pytest.fixture
def marketing(mocker):
    """LinkedinMarketing with a valid token loaded from file."""
    fake_data = {"access_token": "fake_token_123"}
    mocker.patch("builtins.open", mocker.mock_open())
    mocker.patch("json.load", return_value=fake_data)

    from d2b_data.linkedin_marketing import LinkedinMarketing

    return LinkedinMarketing(token_path="fake_token.json")


@pytest.fixture
def marketing_no_file():
    """LinkedinMarketing with no token file specified."""
    from d2b_data.linkedin_marketing import LinkedinMarketing

    return LinkedinMarketing()


@pytest.fixture
def marketing_bad_file(mocker):
    """LinkedinMarketing where the token file cannot be read."""
    mocker.patch("builtins.open", side_effect=Exception("File not found"))

    from d2b_data.linkedin_marketing import LinkedinMarketing

    return LinkedinMarketing(token_path="bad_token.json")
