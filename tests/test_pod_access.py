"""Offline tests for how pods.py reaches a rented pod: SSH key and transfers."""

from __future__ import annotations

import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from miner import pods

ROOT = Path(__file__).resolve().parents[1]


class SshKeyResolutionTests(TestCase):
    """Hardcoding /root/.ssh/id_ed25519 locked out every non-root operator."""

    def resolve(self, home: str, env: dict) -> Path:
        with patch("miner.pods.Path.home", return_value=Path(home)), \
                patch.dict("os.environ", env, clear=True):
            return pods.ssh_key_path()

    def write_lium_config(self, home: str, body: str) -> None:
        config = Path(home) / ".lium"
        config.mkdir(parents=True, exist_ok=True)
        (config / "config.ini").write_text(body, encoding="utf-8")

    def test_defaults_to_the_users_own_key_not_roots(self):
        with TemporaryDirectory() as home:
            self.assertEqual(self.resolve(home, {}), Path(home) / ".ssh/id_ed25519")

    def test_env_override_wins(self):
        with TemporaryDirectory() as home:
            self.write_lium_config(home, "[ssh]\nssh_key_path = /from/config\n")
            self.assertEqual(
                self.resolve(home, {"LIUM_SSH_KEY": "/explicit/key"}),
                Path("/explicit/key"),
            )

    def test_falls_back_to_the_key_lium_already_uses(self):
        with TemporaryDirectory() as home:
            self.write_lium_config(home, "[ssh]\nssh_key_path = /from/config\n")
            self.assertEqual(self.resolve(home, {}), Path("/from/config"))

    def test_user_home_is_expanded_in_both_sources(self):
        with TemporaryDirectory() as home:
            self.write_lium_config(home, "[ssh]\nssh_key_path = ~/keys/lium\n")
            self.assertEqual(
                self.resolve(home, {}).parts[-2:], ("keys", "lium")
            )
            self.assertEqual(
                self.resolve(home, {"LIUM_SSH_KEY": "~/keys/env"}).parts[-2:],
                ("keys", "env"),
            )

    def test_a_malformed_lium_config_falls_back_instead_of_crashing(self):
        with TemporaryDirectory() as home:
            self.write_lium_config(home, "this is not an ini file {{{\n")
            self.assertEqual(self.resolve(home, {}), Path(home) / ".ssh/id_ed25519")

    def test_a_config_without_a_key_path_falls_back(self):
        with TemporaryDirectory() as home:
            self.write_lium_config(home, "[api]\ntoken = abc\n")
            self.assertEqual(self.resolve(home, {}), Path(home) / ".ssh/id_ed25519")

    def test_blank_env_value_is_not_treated_as_a_path(self):
        with TemporaryDirectory() as home:
            self.assertEqual(
                self.resolve(home, {"LIUM_SSH_KEY": "   "}),
                Path(home) / ".ssh/id_ed25519",
            )


class TransferTests(TestCase):
    def test_rsync_is_neither_used_nor_installed(self):
        """Minimal images ship without rsync and often cannot install it."""
        source = (ROOT / "miner/pods.py").read_text()
        self.assertNotIn("apt-get install", source)
        # Scan the code, not the prose that explains why rsync is gone.
        tree = ast.parse(source)
        code = ast.unparse(ast.Module(
            body=[node for node in tree.body
                  if not (isinstance(node, ast.Expr)
                          and isinstance(node.value, ast.Constant)
                          and isinstance(node.value.value, str))],
            type_ignores=[],
        ))
        self.assertNotIn("rsync", code)

    def test_transfers_go_over_tar_and_ssh(self):
        source = (ROOT / "miner/pods.py").read_text()
        for fragment in ("tar -cf -", "tar -xf -"):
            self.assertIn(fragment, source)
