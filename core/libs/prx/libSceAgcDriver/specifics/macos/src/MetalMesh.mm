// Mesh render pipelines (B6 stage 5, owner SHD4; MetalMesh.h): a RenderPipelineRequest whose
// `mesh` is set holds an MSL mesh function in `vertex` and becomes an MTL4MeshRenderPipelineDescriptor
// (no object function). Mesh pipelines are compiled through the device's MTL4Compiler; the pipeline
// archive does not serve them yet (TechnicalDebt metal-mesh-functions).

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "MetalPipelineCache.h"
#include "MetalMesh.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

namespace AgcDriver::Graphics::Metal {

id<MTLRenderPipelineState> PipelineCache::buildMeshRender(const RenderPipelineRequest& request) {
    @autoreleasepool {
        if (request.depthFormat != MTLPixelFormatInvalid && request.stencilFormat != MTLPixelFormatInvalid && request.stencilFormat != request.depthFormat) {
            throw std::runtime_error("Metal pipeline cache (work package D5): separate depth and stencil formats are not supported");
        }
        MTL4MeshRenderPipelineDescriptor* descriptor = [MTL4MeshRenderPipelineDescriptor new];
        descriptor.meshFunctionDescriptor = functionDescriptor(request.vertex);
        // Tessellated draws (MetalTessellation.h): local + hull programs in the object function.
        if (request.object) {
            descriptor.objectFunctionDescriptor = functionDescriptor(*request.object);
            descriptor.maxTotalThreadsPerObjectThreadgroup = request.objectThreads;
        }
        if (request.fragment) descriptor.fragmentFunctionDescriptor = functionDescriptor(*request.fragment);
        for (std::size_t index = 0; index < request.colors.size(); ++index) {
            const auto& color = request.colors[index];
            if (color.format == MTLPixelFormatInvalid) continue;
            MTL4RenderPipelineColorAttachmentDescriptor* attachment = descriptor.colorAttachments[index];
            attachment.pixelFormat = color.format;
            attachment.writeMask = color.writeMask;
            attachment.blendingState = color.blending ? MTL4BlendStateEnabled : MTL4BlendStateDisabled;
            if (color.blending) {
                attachment.sourceRGBBlendFactor = color.sourceRgb;
                attachment.destinationRGBBlendFactor = color.destinationRgb;
                attachment.rgbBlendOperation = color.rgbOperation;
                attachment.sourceAlphaBlendFactor = color.sourceAlpha;
                attachment.destinationAlphaBlendFactor = color.destinationAlpha;
                attachment.alphaBlendOperation = color.alphaOperation;
            }
        }
        descriptor.rasterSampleCount = request.rasterSampleCount;
        descriptor.alphaToCoverageState = request.alphaToCoverage ? MTL4AlphaToCoverageStateEnabled : MTL4AlphaToCoverageStateDisabled;
        descriptor.alphaToOneState = request.alphaToOne ? MTL4AlphaToOneStateEnabled : MTL4AlphaToOneStateDisabled;
        descriptor.rasterizationEnabled = request.rasterization;
        if (request.meshThreads != 0u) descriptor.maxTotalThreadsPerMeshThreadgroup = request.meshThreads;
        // An object stage launches at most meshGroups mesh threadgroups per object threadgroup.
        descriptor.maxTotalThreadgroupsPerMeshGrid = request.object ? request.meshGroups : MaxMeshThreadgroupsPerGrid;
        NSError* error = nil;
        id<MTLRenderPipelineState> pipeline = nil;
        {
            std::shared_lock gate(serializerGate); // MetalPipelineCache.h: no archive flush during the capture
            pipeline = [compiler newRenderPipelineStateWithDescriptor:descriptor compilerTaskOptions:nil error:&error];
        }
        if (pipeline == nil) throw std::runtime_error("Metal device: mesh render pipeline creation failed: " + std::string(error != nil ? error.localizedDescription.UTF8String : "unknown error"));
        {
            std::lock_guard lock(mutex);
            ++statistics.pipelinesCompiled;
        }
        return pipeline;
    }
}

}
