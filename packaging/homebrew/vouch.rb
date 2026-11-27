class Vouch < Formula
  desc "Local authentication multiplexer and process-supervised gateway for AI orchestrators"
  homepage "https://github.com/ayoubzulfiqar/vouch"
  version "3.0.0"
  license "GPL-3.0-or-later"

  on_macos do
    on_arm do
      url "https://github.com/ayoubzulfiqar/vouch/releases/download/v#{version}/vouch-v#{version}-aarch64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_AARCH64_APPLE_DARWIN_SHA256"
    end
    on_intel do
      url "https://github.com/ayoubzulfiqar/vouch/releases/download/v#{version}/vouch-v#{version}-x86_64-apple-darwin.tar.gz"
      sha256 "REPLACE_WITH_X86_64_APPLE_DARWIN_SHA256"
    end
  end

  on_linux do
    on_arm do
      url "https://github.com/ayoubzulfiqar/vouch/releases/download/v#{version}/vouch-v#{version}-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "REPLACE_WITH_AARCH64_UNKNOWN_LINUX_GNU_SHA256"
    end
    on_intel do
      url "https://github.com/ayoubzulfiqar/vouch/releases/download/v#{version}/vouch-v#{version}-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "REPLACE_WITH_X86_64_UNKNOWN_LINUX_GNU_SHA256"
    end
  end

  def install
    bin.install "vouch"
    pkgshare.install "vouch.example.toml"
  end

  def caveats
    <<~EOS
      Scaffold a local config with:
        vouch init --output vouch.toml
      Or inspect detected CLIs and daemons:
        vouch doctor --config vouch.toml
    EOS
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/vouch --version")
  end
end
