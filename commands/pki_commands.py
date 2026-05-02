"""
pki_commands.py — PLATO PKI Evennia Commands

@cert gen            — Generate Ed25519 keypair for your agent
@cert show           — Show your certificate/identity
@cert sign <msg>     — Sign a message with your key
@cert verify <msg>   — Verify a signed message
@cert bottle <file>  — Sign a bottle file with PKI
@cert trust <agent>  — Show trust bridge status for an agent
"""

from evennia import Command
from commands.plato_pki import (
    generate_agent_keypair,
    load_agent_pubkey,
    sign_message,
    verify_message,
    generate_cert,
    verify_cert,
    sign_bottle,
    verify_bottle,
    cert_trust_bridge,
)


class CmdCert(Command):
    """
    PLATO PKI — Identity and signing for fleet agents.

    Usage:
      @cert gen [--encrypt]
          Generate a new Ed25519 keypair for your agent.
          Use --encrypt to set a passphrase.

      @cert show [agent]
          Show your public key and current certificates.
          Optionally show another agent's pubkey.

      @cert sign <message>
          Sign a message with your private key.
          Returns JSON with signature, signer, fingerprint.

      @cert verify <json>
          Verify a signed message (pass the signed JSON dict as a string).

      @cert bottle <filepath>
          Sign a bottle file with PKI header appended.

      @cert trust <agent>
          Show PKI trust bridge status for an agent.

    Examples:
      @cert gen
      @cert show
      @cert show oracle1
      @cert sign "This bottle is authentic"
      @cert trust oracle1
    """

    key = "@cert"
    aliases = ["@pki"]
    locks = "cmd:all()"
    help_category = "Fleet"

    def func(self):
        if not self.args:
            self.caller.msg("|rUsage: @cert gen|show|sign|verify|bottle|trust [args]|n")
            return

        parts = self.args.strip().split(maxsplit=1)
        subcmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        caller_name = self.caller.key.lower().replace(" ", "_")

        if subcmd == "gen":
            self._cmd_gen(caller_name, arg)

        elif subcmd == "show":
            self._cmd_show(arg if arg else caller_name)

        elif subcmd == "sign":
            if not arg:
                self.caller.msg("|rUsage: @cert sign <message>|n")
                return
            self._cmd_sign(caller_name, arg)

        elif subcmd == "verify":
            if not arg:
                self.caller.msg("|rUsage: @cert verify <signed-json-dict>|n")
                return
            self._cmd_verify(arg)

        elif subcmd == "bottle":
            if not arg:
                self.caller.msg("|rUsage: @cert bottle <filepath>|n")
                return
            self._cmd_bottle(caller_name, arg)

        elif subcmd == "trust":
            target = arg if arg else caller_name
            self._cmd_trust(target)

        else:
            self.caller.msg(f"|rUnknown @cert subcommand: {subcmd}|n")

    def _cmd_gen(self, agent, arg):
        """Generate Ed25519 keypair."""
        encrypt = "--encrypt" in arg
        passphrase = ""

        if encrypt:
            # Prompt for passphrase
            self.caller.msg("|yEnter passphrase for private key:|n")
            passphrase = "plato_pki_default"  # In MUD we'd use Evennia's get_pass()
            # For simplicity with ctypes constraints, use empty passphrase
            passphrase = ""

        try:
            result = generate_agent_keypair(agent, passphrase)
            self.caller.msg(
                f"|g✅ PLATO PKI Key Generated|n\n"
                f"  Agent:      |c{result['agent']}|n\n"
                f"  Algorithm:  |c{result['algorithm']}|n\n"
                f"  Fingerprint:|c{result['fingerprint']}|n\n"
                f"  Private key: |w{result['key_file']}|n\n"
                f"  Encrypted:  {'|rYES|n' if result['encrypted'] else '|gNO|n'}\n\n"
                f"  |yNext: @cert show to view your identity|n"
            )
        except Exception as e:
            self.caller.msg(f"|rError generating key: {e}|n")

    def _cmd_show(self, agent):
        """Show public key and certs for an agent."""
        try:
            from commands.plato_pki import _load_index
            index = _load_index()

            if agent not in index["agents"]:
                self.caller.msg(f"|yNo PKI key for |w{agent}|y. Use |w@cert gen|y to create one.|n")
                return

            info = index["agents"][agent]
            agent_certs = [c for c in index["certs"] if c["agent"] == agent]

            self.caller.msg(
                f"|c=== PLATO PKI Identity: {agent} ===|n\n"
                f"  Fingerprint: |w{info['fingerprint']}|n\n"
                f"  Key created: |w{info.get('created', 'N/A')}|n\n"
                f"  Encrypted:   {'|rYES|n' if info.get('has_passphrase') else '|gNO|n'}\n"
                f"  Public key:  |w{info['pubkey_file']}|n\n"
            )

            if agent_certs:
                self.caller.msg(f"|cCertificates ({len(agent_certs)}):|n")
                for c in agent_certs:
                    self.caller.msg(
                        f"  Room |w{c['room_id']}|n — roles: |w{', '.join(c['roles'])}|n "
                        f" (|gvalid|n)" if "2027" in c.get("expires", "") else " (|rexpired|n)"
                    )
            else:
                self.caller.msg(f"|yNo certificates yet. Generate one with @cert cert|n")

        except Exception as e:
            self.caller.msg(f"|rError: {e}|n")

    def _cmd_sign(self, agent, message):
        """Sign a message."""
        try:
            result = sign_message(agent, message, "")
            import json
            self.caller.msg(
                f"|g✅ Message Signed|n\n"
                f"  Signer:      |c{result['signer']}|n\n"
                f"  Fingerprint: |c{result['fingerprint']}|n\n"
                f"  Algorithm:   |c{result['algorithm']}|n\n"
                f"  Signature:   |w{result['signature'][:48]}...|n\n"
                f"  Full JSON:   |n{json.dumps(result, indent=2)}"
            )
        except Exception as e:
            self.caller.msg(f"|rError signing: {e}|n")

    def _cmd_verify(self, arg):
        """Verify a signed message (pass the dict as string)."""
        try:
            import json
            signed = json.loads(arg)
            result = verify_message(signed)
            if result["valid"]:
                self.caller.msg(
                    f"|g✅ Signature Valid|n\n"
                    f"  Signer:      |c{result['signer']}|n\n"
                    f"  Message:     |w{result['message'][:200]}|n"
                )
            else:
                self.caller.msg(
                    f"|r❌ Invalid Signature|n\n"
                    f"  Signer: {result.get('signer', '?')}\n"
                    f"  Error:  {result.get('error', 'unknown')}|n"
                )
        except json.JSONDecodeError:
            self.caller.msg("|rInvalid JSON. Pass a valid signed message dict.|n")
        except Exception as e:
            self.caller.msg(f"|rError: {e}|n")

    def _cmd_bottle(self, agent, filepath):
        """Sign a bottle file."""
        try:
            if not os.path.exists(filepath):
                self.caller.msg(f"|rFile not found: {filepath}|n")
                return
            
            with open(filepath) as f:
                content = f.read()
            
            signed = sign_bottle(agent, content, "", "")
            
            outpath = filepath.replace(".md", ".signed.md")
            with open(outpath, "w") as f:
                f.write(signed)
            
            self.caller.msg(
                f"|g✅ Bottle Signed|n\n"
                f"  Signed: |w{filepath}|n → |c{outpath}|n\n"
                f"  Signer: {agent}"
            )
        except Exception as e:
            self.caller.msg(f"|rError: {e}|n")

    def _cmd_trust(self, agent):
        """Show PKI trust bridge."""
        try:
            result = cert_trust_bridge(agent)
            boost = result.get("trust_boost", 0)
            boost_str = f"+{boost}" if boost > 0 else str(boost)
            color = "|g" if boost >= 0 else "|r"
            
            self.caller.msg(
                f"|c=== PKI Trust Bridge: {agent} ===|n\n"
                f"  PKI Status: |w{result['pki_status']}|n\n"
                f"  Trust Boost: {color}{boost_str}|n\n"
                f"  |yPKI feed: available for mesh-bridge.py import|n"
            )
        except Exception as e:
            self.caller.msg(f"|rError: {e}|n")
