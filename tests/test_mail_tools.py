"""Tests for MailToolKit IMAP/SMTP operations, folder resolution, and draft staging."""

from unittest.mock import MagicMock, patch

import pytest

from config import WorkOSConfig
from workos_engine.tools.mail_tools import MailToolKit


def test_resolve_folder_name_gmail():
    """Verify standard folder aliases resolve correctly for Gmail IMAP."""
    config = WorkOSConfig(imap_host="imap.gmail.com")
    toolkit = MailToolKit(config=config)

    assert toolkit.resolve_folder_name("Drafts") == "[Gmail]/Drafts"
    assert toolkit.resolve_folder_name("draft") == "[Gmail]/Drafts"
    assert toolkit.resolve_folder_name("[Gmail]/Drafts") == "[Gmail]/Drafts"
    assert toolkit.resolve_folder_name("Sent") == "[Gmail]/Sent Mail"
    assert toolkit.resolve_folder_name("sent mail") == "[Gmail]/Sent Mail"
    assert toolkit.resolve_folder_name("Trash") == "[Gmail]/Bin"
    assert toolkit.resolve_folder_name("bin") == "[Gmail]/Bin"
    assert toolkit.resolve_folder_name("Spam") == "[Gmail]/Spam"
    assert toolkit.resolve_folder_name("junk") == "[Gmail]/Spam"
    assert toolkit.resolve_folder_name("All Mail") == "[Gmail]/All Mail"
    assert toolkit.resolve_folder_name("all") == "[Gmail]/All Mail"
    assert toolkit.resolve_folder_name("INBOX") == "INBOX"
    assert toolkit.resolve_folder_name("Custom/Folder") == "Custom/Folder"


def test_resolve_folder_name_standard_imap_with_discovery():
    """Verify fallback discovery for non-Gmail IMAP servers using RFC 6154 flags."""
    config = WorkOSConfig(imap_host="mail.customserver.com")
    toolkit = MailToolKit(config=config)

    mock_mailbox = MagicMock()
    mock_f1 = MagicMock()
    mock_f1.name = "INBOX.Drafts"
    mock_f1.flags = ("\\Drafts", "\\HasNoChildren")
    mock_f2 = MagicMock()
    mock_f2.name = "INBOX.Sent"
    mock_f2.flags = ("\\Sent", "\\HasNoChildren")
    mock_mailbox.folder.list.return_value = [mock_f1, mock_f2]

    # With discovery via mailbox instance
    resolved = toolkit.resolve_folder_name("Drafts", mailbox=mock_mailbox)
    assert resolved == "INBOX.Drafts"

    resolved_sent = toolkit.resolve_folder_name("Sent", mailbox=mock_mailbox)
    assert resolved_sent == "INBOX.Sent"


def test_create_draft_uses_resolved_folder():
    """Verify create_draft resolves folder alias and appends message."""
    config = WorkOSConfig(
        imap_host="imap.gmail.com",
        imap_user="test@gmail.com",
        imap_password="password",
        smtp_user="test@gmail.com",
    )
    toolkit = MailToolKit(config=config)

    with patch.object(toolkit, "_get_mailbox") as mock_get_mb:
        mock_mb = MagicMock()
        mock_get_mb.return_value.__enter__.return_value = mock_mb

        res = toolkit.create_draft(
            to_email="colleague@example.com",
            subject="Quarterly Review",
            body="Attached is the summary.",
            folder="Drafts",
        )

        assert res["status"] == "draft_created"
        assert res["staged_as_draft"] is True
        mock_get_mb.assert_called_with(folder="[Gmail]/Drafts")
        assert mock_mb.append.called
        call_kwargs = mock_mb.append.call_args
        assert call_kwargs[1]["folder"] == "[Gmail]/Drafts" or call_kwargs[0][1] == "[Gmail]/Drafts"


def test_search_emails_resolves_drafts_folder():
    """Verify search_emails in Drafts targets [Gmail]/Drafts and catches nonexistent gracefully."""
    config = WorkOSConfig(
        imap_host="imap.gmail.com",
        imap_user="test@gmail.com",
        imap_password="password",
    )
    toolkit = MailToolKit(config=config)

    with patch.object(toolkit, "_get_mailbox") as mock_get_mb:
        mock_mb = MagicMock()
        mock_mb.fetch.return_value = []
        mock_get_mb.return_value.__enter__.return_value = mock_mb

        results = toolkit.search_emails(folder="Drafts", text="")
        assert results == []
        mock_get_mb.assert_called_with(folder="[Gmail]/Drafts")


def test_live_gmail_imap_drafts_resolution():
    """Integration test against configured Gmail account ensuring no NONEXISTENT error."""
    config = WorkOSConfig()
    if not config.imap_host or not config.imap_user or not config.imap_password:
        pytest.skip("IMAP credentials not configured in environment")

    toolkit = MailToolKit(config=config)
    try:
        results = toolkit.search_emails(folder="Drafts", limit=5)
        assert isinstance(results, list)
    except Exception as e:
        pytest.fail(f"search_emails in Drafts raised unexpected exception: {e}")
