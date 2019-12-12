

#pragma once


namespace Simulation
{
    template <typename MeshT, typename OptionT>
    void step(MeshT& mesh, const OptionT& opts) {
        if (opts.integrator == 0) {
            XPBD::step(mesh, opts);
        }
    }
}
