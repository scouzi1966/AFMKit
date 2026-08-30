// FLUX.2 VAE — decoder path, Swift/MLX port.
//
// Isomorphic to mflux's flux2_vae (the implementation the Python lens-mlx port loads;
// VAE gate there: 57.65 dB vs PT). Decoder-only: Lens t2i never encodes. Tensors flow
// NCHW between blocks with NHWC transposes around convs/norms — kept identical to the
// reference so the two sources diff cleanly (isomorphism rule), at a negligible cost
// for a one-shot decode.
//
// The `bn` running stats implement the T1 latent de-norm in PATCHIFIED space:
// x*std + mean, then unpatchify, then decode.

import Foundation
import MLX
import MLXFast
import MLXNN

public final class Flux2BatchNormStats: Module {
    @ParameterInfo(key: "running_mean") public var runningMean: MLXArray
    @ParameterInfo(key: "running_var") public var runningVar: MLXArray
    public let eps: Float

    public init(numFeatures: Int, eps: Float = 1e-4) {
        self._runningMean.wrappedValue = MLXArray.zeros([numFeatures], dtype: .float32)
        self._runningVar.wrappedValue = MLXArray.ones([numFeatures], dtype: .float32)
        self.eps = eps
        super.init()
    }
}

public final class Flux2ResnetBlock2D: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

    public init(inChannels: Int, outChannels: Int, eps: Float = 1e-6, groups: Int = 32) {
        self._norm1.wrappedValue = GroupNorm(
            groupCount: groups, dimensions: inChannels, eps: eps, pytorchCompatible: true)
        self._conv1.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: outChannels,
            kernelSize: 3, stride: 1, padding: 1)
        self._norm2.wrappedValue = GroupNorm(
            groupCount: groups, dimensions: outChannels, eps: eps, pytorchCompatible: true)
        self._conv2.wrappedValue = Conv2d(
            inputChannels: outChannels, outputChannels: outChannels,
            kernelSize: 3, stride: 1, padding: 1)
        self._convShortcut.wrappedValue = inChannels != outChannels
            ? Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                     kernelSize: 1, stride: 1)
            : nil
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let dtype = hiddenStates.dtype
        var residual = hiddenStates.transposed(0, 2, 3, 1)

        var h = hiddenStates.transposed(0, 2, 3, 1)
        h = norm1(h.asType(.float32)).asType(dtype)
        h = silu(h)
        h = conv1(h)
        h = norm2(h.asType(.float32)).asType(dtype)
        h = silu(h)
        h = conv2(h)

        if let convShortcut {
            residual = convShortcut(residual)
        }
        return (h + residual).transposed(0, 3, 1, 2)
    }
}

public final class Flux2AttentionBlock: Module {
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    public init(channels: Int, groups: Int = 32, eps: Float = 1e-6) {
        self._groupNorm.wrappedValue = GroupNorm(
            groupCount: groups, dimensions: channels, eps: eps, pytorchCompatible: true)
        self._toQ.wrappedValue = Linear(channels, channels)
        self._toK.wrappedValue = Linear(channels, channels)
        self._toV.wrappedValue = Linear(channels, channels)
        self._toOut.wrappedValue = Linear(channels, channels)
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let dtype = hiddenStates.dtype
        var h = hiddenStates.transposed(0, 2, 3, 1)
        let (batch, height, width, channels) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))

        let normed = groupNorm(h.asType(.float32)).asType(dtype)
        let q = toQ(normed).reshaped(batch, height * width, 1, channels).transposed(0, 2, 1, 3)
        let k = toK(normed).reshaped(batch, height * width, 1, channels).transposed(0, 2, 1, 3)
        let v = toV(normed).reshaped(batch, height * width, 1, channels).transposed(0, 2, 1, 3)

        let scale = 1.0 / sqrt(Float(channels))
        var attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: .none)
        attended = attended.transposed(0, 2, 1, 3).reshaped(batch, height, width, channels)
        h = h + toOut(attended)
        return h.transposed(0, 3, 1, 2)
    }
}

public final class Flux2Upsample2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    public init(channels: Int, outChannels: Int? = nil) {
        self._conv.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: outChannels ?? channels,
            kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = MLX.repeated(hiddenStates, count: 2, axis: 2)
        h = MLX.repeated(h, count: 2, axis: 3)
        h = h.transposed(0, 2, 3, 1)
        h = conv(h)
        return h.transposed(0, 3, 1, 2)
    }
}

public final class Flux2UNetMidBlock2D: Module {
    @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock2D]
    @ModuleInfo(key: "attentions") var attentions: [Flux2AttentionBlock]

    public init(channels: Int, eps: Float = 1e-6, groups: Int = 32) {
        self._resnets.wrappedValue = [
            Flux2ResnetBlock2D(inChannels: channels, outChannels: channels, eps: eps, groups: groups),
            Flux2ResnetBlock2D(inChannels: channels, outChannels: channels, eps: eps, groups: groups),
        ]
        self._attentions.wrappedValue = [Flux2AttentionBlock(channels: channels, groups: groups, eps: eps)]
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = resnets[0](hiddenStates)
        h = attentions[0](h)
        h = resnets[1](h)
        return h
    }
}

public final class Flux2UpDecoderBlock2D: Module {
    @ModuleInfo(key: "resnets") var resnets: [Flux2ResnetBlock2D]
    @ModuleInfo(key: "upsamplers") var upsamplers: [Flux2Upsample2D]

    public init(
        inChannels: Int, outChannels: Int, numLayers: Int = 3,
        eps: Float = 1e-6, groups: Int = 32, addUpsample: Bool = true
    ) {
        self._resnets.wrappedValue = (0..<numLayers).map { i in
            Flux2ResnetBlock2D(
                inChannels: i == 0 ? inChannels : outChannels,
                outChannels: outChannels, eps: eps, groups: groups)
        }
        self._upsamplers.wrappedValue = addUpsample
            ? [Flux2Upsample2D(channels: outChannels, outChannels: outChannels)] : []
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        var h = hiddenStates
        for resnet in resnets {
            h = resnet(h)
            eval(h)  // full-res resnets hold multi-GB of live intermediates
        }
        for upsampler in upsamplers {
            h = upsampler(h)
            eval(h)
        }
        return h
    }
}

public final class Flux2Decoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var midBlock: Flux2UNetMidBlock2D
    @ModuleInfo(key: "up_blocks") var upBlocks: [Flux2UpDecoderBlock2D]
    @ModuleInfo(key: "conv_norm_out") var convNormOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d

    public init(
        inChannels: Int = 32, outChannels: Int = 3,
        blockOutChannels: [Int] = [128, 256, 512, 512],
        layersPerBlock: Int = 2, normNumGroups: Int = 32, eps: Float = 1e-6
    ) {
        self._convIn.wrappedValue = Conv2d(
            inputChannels: inChannels, outputChannels: blockOutChannels.last!,
            kernelSize: 3, stride: 1, padding: 1)
        self._midBlock.wrappedValue = Flux2UNetMidBlock2D(
            channels: blockOutChannels.last!, eps: eps, groups: normNumGroups)

        let reversedChannels = Array(blockOutChannels.reversed())
        self._upBlocks.wrappedValue = reversedChannels.enumerated().map { i, outputChannel in
            Flux2UpDecoderBlock2D(
                inChannels: i == 0 ? outputChannel : reversedChannels[i - 1],
                outChannels: outputChannel,
                numLayers: layersPerBlock + 1,
                eps: eps, groups: normNumGroups,
                addUpsample: i != reversedChannels.count - 1)
        }
        self._convNormOut.wrappedValue = GroupNorm(
            groupCount: normNumGroups, dimensions: blockOutChannels[0],
            eps: eps, pytorchCompatible: true)
        self._convOut.wrappedValue = Conv2d(
            inputChannels: blockOutChannels[0], outputChannels: outChannels,
            kernelSize: 3, stride: 1, padding: 1)
        super.init()
    }

    public func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let dtype = hiddenStates.dtype
        var h = hiddenStates.transposed(0, 2, 3, 1)
        h = convIn(h)
        eval(h)  // stage checkpoints: the whole decode as ONE lazy graph holds ~10-15 GB
                 // of live intermediates at 1024² (the dominant peak in both Lens and
                 // ERNIE pipelines); per-stage eval bounds the high-water cheaply.
        h = h.transposed(0, 3, 1, 2)
        h = midBlock(h)
        eval(h)
        for upBlock in upBlocks {
            h = upBlock(h)
            eval(h)
        }
        h = h.transposed(0, 2, 3, 1)
        h = convNormOut(h.asType(.float32)).asType(dtype)
        h = silu(h)
        h = convOut(h)
        return h.transposed(0, 3, 1, 2)
    }
}

/// Decoder-only FLUX.2 VAE: post_quant_conv + decoder + the `bn` latent de-norm stats.
/// (The reference's encoder/quant_conv are omitted — t2i never encodes; their
/// checkpoint keys are filtered at load.)
public final class Flux2VAE: Module {
    public static let scalingFactor: Float = 1.0
    public static let shiftFactor: Float = 0.0
    public static let latentChannels = 32

    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    @ModuleInfo(key: "decoder") var decoder: Flux2Decoder
    @ModuleInfo(key: "bn") var bn: Flux2BatchNormStats

    public override init() {
        self._postQuantConv.wrappedValue = Conv2d(
            inputChannels: Self.latentChannels, outputChannels: Self.latentChannels,
            kernelSize: 1, padding: 0)
        self._decoder.wrappedValue = Flux2Decoder()
        self._bn.wrappedValue = Flux2BatchNormStats(numFeatures: 4 * Self.latentChannels)
        super.init()
    }

    public func decode(_ latents: MLXArray) -> MLXArray {
        // scaling_factor 1.0 / shift_factor 0.0 — kept for isomorphism.
        var l = (latents / Self.scalingFactor) + Self.shiftFactor
        l = l.transposed(0, 2, 3, 1)
        l = postQuantConv(l)
        l = l.transposed(0, 3, 1, 2)
        return decoder(l)
    }

    /// T1: bn de-norm in patchified space (x*std + mean), unpatchify, decode.
    public func decodePackedLatents(_ packedLatents: MLXArray) -> MLXArray {
        let bnMean = bn.runningMean.reshaped(1, -1, 1, 1)
        let bnStd = sqrt(bn.runningVar.reshaped(1, -1, 1, 1) + bn.eps)
        var latents = packedLatents * bnStd.asType(packedLatents.dtype)
            + bnMean.asType(packedLatents.dtype)
        latents = Self.unpatchifyLatents(latents)
        return decode(latents)
    }

    /// (T4) [B, C, H, W] with C = 4*c -> [B, c, 2H, 2W], diffusers patch order.
    static func unpatchifyLatents(_ latents: MLXArray) -> MLXArray {
        let (b, c, h, w) = (latents.dim(0), latents.dim(1), latents.dim(2), latents.dim(3))
        var x = latents.reshaped(b, c / 4, 2, 2, h, w)
        x = x.transposed(0, 1, 4, 2, 5, 3)
        return x.reshaped(b, c / 4, h * 2, w * 2)
    }
}
