//#pragma once
//
//#include <memory>
//#include <vector>
//
//namespace alus {
//namespace snapengine {
//namespace math{
//
//
//class Range{
///**
//     * Computes the value range for the values in the given <code>float</code> array. Values at a given index
//     * <code>i</code> for which <code>validator.validate(i)</code> returns <code>false</code> are excluded from the
//     * computation.
//     *
//     * @param values    the array whose value range to compute
//     * @param validator used to validate the array indexes, must not be <code>null</code>. Use {@link
//     *                  IndexValidator#TRUE} instead.
//     * @param range     if not <code>null</code>, used as return value, otherwise a new instance is created
//     * @param pm        a monitor to inform the user about progress
//     *
//     * @return the value range for the given array
//     */
//public:
// static std::shared_ptr<Range> ComputeRangeFloat(std::vector<float> values,
////                                         IndexValidator validator,
//                                                 bool validator,
//                                         std::shared_ptr<Range> range/*,
//                                         ProgressMonitor pm*/);
//};
//
//} // namespace math
//}  // namespace snapengine
//}  // namespace alus
