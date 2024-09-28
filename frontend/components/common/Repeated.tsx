import { forwardRefWithAs } from "@/utils/forwardRefWithAs";

interface Props {
  times: number;
}

export const Repeated = forwardRefWithAs<"div", Props>((props, ref) => {
  const { as: Tag = "div", times, ...rest } = props;

  return Array.from({ length: times }).map((_, index) => (
    <Tag ref={ref} {...rest} key={index} />
  ));
});
